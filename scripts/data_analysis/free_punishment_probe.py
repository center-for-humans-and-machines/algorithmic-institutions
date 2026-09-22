"""Is a punishment aimed at a player who gave no input free, and is it shown?
(auto/free-punishment-fix, step 1 -- run before and after the fix)

`ArtificialHumanEnv.punish` stores the manager's action verbatim for all eight
agents, including the ones the validity model has just marked as timed out.
The accounting then discards it -- `compute_common_good_per_group` zeroes
punishment at `~contribution_valid`, `compute_payoff_per_group` zeroes the
whole invalid contributor's payoff -- so nothing is charged for it. But
`step()` copies the raw value into `prev_punishment`, which is the contribution
artificial human's only channel from the manager, and `served_state()` corrects
only `contribution` and `prev_contribution`. Free deterrence, on a cell every
model can identify. In the human data all 560 timed-out rows carry punishment
exactly 0.

Two independent measurements, both run before and after the fix:

`--mode env` (plain torch, no PyG, runs anywhere)
    The counterfactual. One round with agent 0 timed out, played once with 0
    and once with the maximum punishment on that agent, everything else held:
    the group's common good and payoff sum, the env's recorded punishment, and
    what the contribution and switch models are served on their punishment
    channels. Identical reward with a different input to every downstream model
    is the lever; after the fix the served and recorded values must be 0 in
    both arms while the reward is unmoved.

`--mode sim --config <yml>` (needs torch_geometric to unpickle the GNNs)
    How often the path fires in the real stack: the timeout rate over
    agent-rounds, how many of those cells the artificial punisher actually
    spends a nonzero punishment on, the value distribution it spends, and --
    recomputed from the env's own accounting on every round -- the largest
    change in common good and in group payoff that zeroing those cells would
    cause. It also reports what each model is served on its punishment channel
    at those cells and what the recorded state carries there, which is the one
    thing this fix changes about `per_round.parquet`.

Usage:
    PYTHONPATH=src python scripts/data_analysis/free_punishment_probe.py \\
        --mode env --out plots/data_analysis/evaluation/free_punishment/env_before.json
    python scripts/data_analysis/free_punishment_probe.py --mode sim \\
        --config configs/simulation/manager_testing/<config>.yml \\
        --episodes 20 --out plots/data_analysis/evaluation/free_punishment/sim_before.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import count

import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.manager.environment import ArtificialHumanEnv  # noqa: E402

# the punishment channels, with the validity flag that says a cell is a
# timeout rather than a real decision. `punishment` is round t's own, read by
# the switch model; `prev_punishment` is round t-1's, the contribution
# model's only channel from the manager.
AUDITED = {
    "punishment": "contribution_valid",
    "prev_punishment": "prev_contribution_valid",
}

AGENT_GROUP = [0, 0, 0, 0, 1, 1, 1, 1]
DEFAULTS = {
    "punishment": 0,
    "contribution": 9,
    "contribution_valid": False,
    "punishment_valid": False,
    "common_good": 12.333333333333334,
    "agent_group": 0,
    "does_switch": False,
    "own_grp_prev_mean_contr": 9,
}
# agent 0 times out, agent 1 genuinely contributes 0
C_ROUND = [7, 0, 20, 3, 11, 5, 14, 2]
VALID = [False] + [True] * 7


class _Fixed:
    """Artificial human stand-in returning the same draw every round."""

    autoregressive = False
    default_values = DEFAULTS

    def __init__(self, row, dtype=th.int64):
        self.row, self.dtype = row, dtype

    def predict(self, state, **_):
        return th.tensor(self.row, dtype=self.dtype).reshape(1, 8, 1), None


def _mock_env(n_rounds=3):
    return ArtificialHumanEnv(
        artifical_humans=_Fixed(C_ROUND),
        artifical_humans_valid=_Fixed(VALID, dtype=th.bool),
        artifical_humans_switch=None,
        switch_every=4,
        batch_size=1,
        n_agents=8,
        n_contributions=21,
        n_punishments=31,
        n_rounds=n_rounds,
        device=th.device("cpu"),
        n_groups=2,
        agent_groups=AGENT_GROUP,
        default_values=DEFAULTS,
    )


def env_mode():
    """One round, agent 0 timed out, the manager playing 0 then 30 on it."""
    arms = {}
    for label, aimed in (("plays_0", 0), ("plays_30", 30)):
        env = _mock_env()
        env.reset()
        action = th.ones((1, 8, 1), dtype=th.int64)  # everyone else punished 1
        action[0, 0, 0] = aimed
        state = env.punish(action)
        cg = float(state["common_good"].reshape(-1)[0])
        payoff_sum = float(env.group_payoff_sum.reshape(-1)[0])
        recorded = float(state["punishment"].reshape(-1)[0])
        served_now = float(env.served_state()["punishment"].reshape(-1)[0])
        env.step()
        served_prev = float(env.served_state()["prev_punishment"].reshape(-1)[0])
        arms[label] = {
            "aimed_at_timed_out_agent": aimed,
            "group_common_good": round(cg, 6),
            "group_payoff_sum": round(payoff_sum, 6),
            "env_recorded_punishment": recorded,
            "served_punishment_same_round": served_now,
            "served_prev_punishment_next_round": served_prev,
        }
    a, b = arms["plays_0"], arms["plays_30"]
    return {
        "mode": "env",
        "arms": arms,
        "reward_identical": (
            a["group_common_good"] == b["group_common_good"]
            and a["group_payoff_sum"] == b["group_payoff_sum"]
        ),
        "served_differs": (
            a["served_prev_punishment_next_round"]
            != b["served_prev_punishment_next_round"]
            or a["served_punishment_same_round"] != b["served_punishment_same_round"]
        ),
        "recorded_differs": (
            a["env_recorded_punishment"] != b["env_recorded_punishment"]
        ),
    }


class Probe:
    """Wrap a model's predict and count what it is served at timeout cells."""

    def __init__(self, label, model):
        self.label, self.calls = label, 0
        self.reads = _read_keys(model)
        self.served = defaultdict(Counter)
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


def _read_keys(model):
    """The state keys the model's encoder consumes."""
    if hasattr(model, "features"):  # LinearAHAdapter
        return sorted(model.features)
    keys = [e["name"] for e in getattr(model, "x_encoding", [])]
    keys += [e["name"] for e in getattr(model, "u_encoding", []) or []]
    keys += [e["name"] for e in getattr(model, "b_encoding", []) or []]
    return sorted(set(keys))


def sim_mode(config_path, episodes):
    import numpy as np
    import yaml

    from aimanager.manager.api_manager import MultiManager
    from aimanager.simulation.linear_ah import load_ah_model
    from aimanager.simulation.simulate import add_punishments, make_round

    with open(config_path) as fh:
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

    ahc = cfg["artificial_humans"]["group_switching"]

    def load(key):
        return load_ah_model(
            os.path.join(basedir, ahc[key]),
            device=device,
            n_agents=n_agents,
            n_contributions=cfg["n_contributions"],
        )

    ah, ah_val = load("contribution_model"), load("valid_model")
    ah_switch = load("switch_model") if "switch_model" in ahc else None
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
    n_cells = timeouts = free_spent = 0
    spent = Counter()
    recorded_p = Counter()
    pun_served = Counter()
    max_d_cg = max_d_payoff = 0.0
    for e in range(episodes):
        state = env.reset()
        rounds = []
        for round_number in count():
            ag = state["agent_group"].squeeze().tolist()
            cv = state["contribution_valid"].reshape(-1).tolist()
            contributions = state["contribution"].squeeze().tolist()
            n_cells += len(cv)
            timeouts += sum(1 for v in cv if not v)
            rd = make_round(
                contributions,
                round_number,
                [group_map[g] for g in ag],
                e,
                agent_group=ag,
                contribution_valid=cv,
            )
            # what the punisher itself is served on its own `prev_punishment`
            # feature: `api_manager.create_data` shifts the previous round
            # record, so a cell that timed out then is read here
            if rounds:
                prev = rounds[-1]
                for value, ok in zip(prev["punishment"], prev["contribution_valid"]):
                    if not ok:
                        pun_served[int(value)] += 1
            p = mm.get_punishments(rounds + [rd])[0]
            for value, ok in zip(p, cv):
                if not ok:
                    spent[int(value)] += 1
                    free_spent += int(value) > 0
            p_raw = th.tensor(p, dtype=th.int64, device=device)
            p_raw = p_raw.unsqueeze(-1).unsqueeze(0)
            valid = state["contribution_valid"]
            p_zeroed = th.where(valid, p_raw, th.zeros_like(p_raw))
            # what the manager is charged, both ways, from the env's own
            # accounting -- the claim that the lever is free
            cg_raw = env.compute_common_good_per_group(
                state["contribution"], p_raw, valid
            )
            cg_zero = env.compute_common_good_per_group(
                state["contribution"], p_zeroed, valid
            )
            max_d_cg = max(max_d_cg, float((cg_raw - cg_zero).abs().max()))
            pay_raw = env.compute_payoff_per_group(
                state["contribution"], p_raw, valid, cg_raw.gather(1, env.agent_groups)
            )[2]
            pay_zero = env.compute_payoff_per_group(
                state["contribution"],
                p_zeroed,
                valid,
                cg_zero.gather(1, env.agent_groups),
            )[2]
            max_d_payoff = max(max_d_payoff, float((pay_raw - pay_zero).abs().max()))

            state = env.punish(p_raw)
            charged = state["punishment"].reshape(-1).tolist()
            for value, ok in zip(charged, cv):
                if not ok:
                    recorded_p[int(value)] += 1
            # mirrors simulate.py: the manager's record is the charged value
            rounds.append(add_punishments(rd, charged))
            state, _, done = env.step()
            if done:
                break

    return {
        "mode": "sim",
        "config": config_path,
        "episodes": episodes,
        "agent_rounds": n_cells,
        "timeouts": timeouts,
        "timeout_rate": timeouts / n_cells if n_cells else float("nan"),
        "punisher_spend_at_timeout_cells": {
            str(k): v for k, v in sorted(spent.items())
        },
        "free_punishments": free_spent,
        "free_punishment_rate_of_timeouts": free_spent / timeouts if timeouts else 0.0,
        "free_punishment_rate_of_agent_rounds": (
            free_spent / n_cells if n_cells else 0.0
        ),
        "env_recorded_punishment_at_timeouts": {
            str(k): v for k, v in sorted(recorded_p.items())
        },
        "punisher_served_prev_punishment_at_timeouts": {
            str(k): v for k, v in sorted(pun_served.items())
        },
        "max_common_good_change_if_zeroed": max_d_cg,
        "max_group_payoff_change_if_zeroed": max_d_payoff,
        "models": [p.report() for p in probes],
    }


def run_mode(config_path, out_dir):
    """The scored run itself, counted where the action is applied.

    `--mode sim` rolls the stack out from this file and so does not consume
    the RNG in exactly the order `simulate.py` does; this mode runs
    `run_simulation` unchanged, with `ArtificialHumanEnv.punish` wrapped, and
    therefore reports the rate on the very trajectory that is scored.
    """
    import yaml

    from aimanager.manager.environment import ArtificialHumanEnv
    from aimanager.simulation.simulate import run_simulation

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg = {**cfg, "save_per_round": False}

    stats = {"agent_rounds": 0, "timeouts": 0, "aimed_at_timeouts": Counter()}
    inner = ArtificialHumanEnv.punish

    def counting_punish(self, punishment):
        valid = self.contribution_valid.detach().cpu().reshape(-1).numpy()
        aimed = punishment.detach().cpu().reshape(-1).numpy()
        stats["agent_rounds"] += len(valid)
        stats["timeouts"] += int((~valid.astype(bool)).sum())
        for value in aimed[~valid.astype(bool)]:
            stats["aimed_at_timeouts"][int(value)] += 1
        return inner(self, punishment)

    ArtificialHumanEnv.punish = counting_punish
    try:
        run_simulation(cfg, out_dir)
    finally:
        ArtificialHumanEnv.punish = inner

    free = sum(n for v, n in stats["aimed_at_timeouts"].items() if v > 0)
    return {
        "mode": "run",
        "config": config_path,
        "agent_rounds": stats["agent_rounds"],
        "timeouts": stats["timeouts"],
        "timeout_rate": stats["timeouts"] / stats["agent_rounds"],
        "aimed_at_timeout_cells": {
            str(k): v for k, v in sorted(stats["aimed_at_timeouts"].items())
        },
        "free_punishments": free,
        "free_punishment_rate_of_timeouts": free / stats["timeouts"],
        "free_punishment_rate_of_agent_rounds": free / stats["agent_rounds"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["env", "sim", "run"], default="env")
    ap.add_argument("--config")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--scratch", default="temp/free_punishment_run")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.mode == "env":
        out = env_mode()
    elif args.mode == "run":
        assert args.config, "--mode run needs --config"
        out = run_mode(args.config, args.scratch)
    else:
        assert args.config, "--mode sim needs --config"
        out = sim_mode(args.config, args.episodes)

    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
