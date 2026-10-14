"""What value does each artificial-human model actually receive for a player
who timed out, in a live simulation? (auto/sim-timeout-imputation, step 1)

The environment samples who times out with the configured `valid_model` and
then overwrites those players' contributions with the imputed default
(`environment.update_contribution`) before the state is passed on. The env's
own common-good accounting zeroes them separately and is correct; the defect
is purely in what the models are shown. `auto/punisher-timeout-feature` fixed
the two punisher serving paths; this probe measures the remaining two.

It runs the real simulation machinery (same env, same models, same protocol,
fewer episodes) with every model's `predict` wrapped, and reports per model:

  * the state keys its encoder actually reads (GNN `x_encoding` / linear
    `features`) -- a model that never reads a contribution key cannot be
    affected at all, and that has to be shown rather than assumed;
  * the value it was served at every timed-out cell it reads, as a value
    count (`prev_contribution` where `prev_contribution_valid` is False for
    the prev-anchored contribution model, `contribution` where
    `contribution_valid` is False for the current-anchored switch model);
  * how often the path fires: the realised timeout rate over agent-rounds.

Run it before and after the fix: the served value must go from the imputed
default to 0 and the timeout rate must stay put (the valid model's own inputs
are unchanged).

Usage (Raven, torch_geometric needed to unpickle the GNNs):
    python scripts/data_analysis/sim_timeout_serving_probe.py \\
        configs/simulation/manager_testing/<config>.yml \\
        --episodes 20 --out plots/data_analysis/evaluation/<dir>/<name>.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import count

import numpy as np
import torch as th
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aimanager.manager.api_manager import MultiManager  # noqa: E402
from aimanager.manager.environment import ArtificialHumanEnv  # noqa: E402
from aimanager.simulation.linear_ah import load_ah_model  # noqa: E402
from aimanager.simulation.simulate import add_punishments, make_round  # noqa: E402

# state keys whose served value we audit, with the validity flag that says a
# cell is a timeout rather than a real decision
AUDITED = {
    "contribution": "contribution_valid",
    "prev_contribution": "prev_contribution_valid",
}


def read_keys(model):
    """The state keys the model's encoder consumes."""
    if hasattr(model, "features"):  # LinearAHAdapter
        return sorted(model.features)
    keys = [e["name"] for e in getattr(model, "x_encoding", [])]
    keys += [e["name"] for e in getattr(model, "u_encoding", []) or []]
    keys += [e["name"] for e in getattr(model, "b_encoding", []) or []]
    return sorted(set(keys))


class Probe:
    """Wrap a model's predict and count what it is served at timeout cells."""

    def __init__(self, label, model):
        self.label = label
        self.reads = read_keys(model)
        self.served = defaultdict(Counter)  # audited key -> value counts
        self.calls = 0
        self._model = model
        self._inner = model.predict
        model.predict = self

    def __call__(self, state, **kwargs):
        self.calls += 1
        for key, flag in AUDITED.items():
            if key not in state or flag not in state:
                continue
            v = state[key].detach().cpu().reshape(-1).numpy()
            ok = state[flag].detach().cpu().reshape(-1).numpy().astype(bool)
            for x in v[~ok]:
                self.served[key][float(x)] += 1
        return self._inner(state, **kwargs)

    def report(self):
        return {
            "label": self.label,
            "reads": self.reads,
            "calls": self.calls,
            "served_at_timeout_cells": {
                k: {str(val): n for val, n in sorted(c.items())}
                for k, c in self.served.items()
            },
        }


def punisher_probe(mm):
    """The punisher is served through the manager API, not env.predict; audit
    the round dicts it is handed (both families read `contribution` there)."""
    seen = Counter()
    inner = mm.get_punishments

    def wrapped(rounds, *a, **kw):
        last = rounds[-1]
        for c, ok in zip(last["contribution"], last["contribution_valid"]):
            if not ok:
                seen[float(c)] += 1
        return inner(rounds, *a, **kw)

    mm.get_punishments = wrapped
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    th.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    basedir = cfg.get("basedir", ".")
    device = th.device("cpu")
    n_agents = cfg["n_agents"]

    managers = {
        k: ({**v, "model_path": os.path.join(basedir, v["path"])} if "path" in v else v)
        for k, v in cfg["managers"].items()
    }
    mm = MultiManager(managers, n_steps=cfg["n_episode_steps"])
    pun_served = punisher_probe(mm)

    ahc = cfg["artificial_humans"]["group_switching"]

    def load(key):
        return load_ah_model(
            os.path.join(basedir, ahc[key]),
            device=device,
            n_agents=n_agents,
            n_contributions=cfg["n_contributions"],
        )

    ah, ah_val, ah_switch = load("contribution_model"), load("valid_model"), None
    if "switch_model" in ahc:
        ah_switch = load("switch_model")

    probes = [Probe("contribution", ah), Probe("valid", ah_val)]
    if ah_switch is not None:
        probes.append(Probe("switch", ah_switch))

    env = ArtificialHumanEnv(
        artifical_humans=ah,
        artifical_humans_valid=ah_val,
        artifical_humans_switch=ah_switch,
        switch_every=cfg.get("switch_every"),
        n_agents=n_agents,
        n_contributions=cfg["n_contributions"],
        n_punishments=cfg["n_punishments"],
        n_rounds=cfg["n_rounds"],
        n_groups=cfg["n_groups"],
        batch_size=1,
        device=device,
        agent_groups=cfg.get("agent_groups"),
        reward_mode=cfg.get("reward_mode", "sum"),
    )

    pairing = cfg["pairings"][0]
    group_map = [pairing["group_0"], pairing["group_1"]]
    realised, n_cells, recorded_c = 0, 0, Counter()
    for e in range(args.episodes):
        state = env.reset()
        rounds, idx = [], e
        for round_number in count():
            ag = state["agent_group"].squeeze().tolist()
            cv = state["contribution_valid"].reshape(-1).tolist()
            contributions = state["contribution"].squeeze().tolist()
            n_cells += len(cv)
            realised += sum(1 for v in cv if not v)
            for c, ok in zip(contributions, cv):
                if not ok:
                    recorded_c[float(c)] += 1
            rd = make_round(
                contributions,
                round_number,
                [group_map[g] for g in ag],
                idx,
                agent_group=ag,
                contribution_valid=cv,
            )
            p = mm.get_punishments(rounds + [rd])[0]
            rounds.append(add_punishments(rd, p))
            state = env.punish(
                th.tensor(p, dtype=th.int64, device=device).unsqueeze(-1).unsqueeze(0)
            )
            state, _, done = env.step()
            if done:
                break

    out = {
        "config": args.config,
        "episodes": args.episodes,
        "agent_rounds": n_cells,
        "timeouts": realised,
        "timeout_rate": realised / n_cells if n_cells else float("nan"),
        "env_recorded_contribution_at_timeouts": {
            str(k): v for k, v in sorted(recorded_c.items())
        },
        "punisher_served_at_timeouts": {
            str(k): v for k, v in sorted(pun_served.items())
        },
        "models": [p.report() for p in probes],
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
