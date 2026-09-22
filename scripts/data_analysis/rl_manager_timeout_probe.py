"""What value does the reinforcement-learning manager actually receive for a
player who timed out? (auto/rl-manager-timeout-view, step 1)

`auto/punisher-timeout-feature` corrected the punisher's two serving paths and
`auto/sim-timeout-imputation` corrected the contribution and switch models'
through `ArtificialHumanEnv.served_state`. The RL manager's path was left
alone: `reset()`, `punish()` and `step()` return `self.state`, the raw one, so
`rl_manager.run_batch` hands the manager -- and the replay buffer it trains
from -- the imputed default (9) where the game recorded 0.

This probe drives the **real** training rollout (`rl_manager.run_batch`, the
real env, the real `ArtificalManager`, the real replay `Memory`) and reports,
at every cell whose `contribution_valid` is False:

  * what `manager.get_action` was served, next to what `env.state` records at
    the same moment -- the side-by-side is the confirmation;
  * what was written into the replay buffer, i.e. what the TD update trains on;
  * what the fixed opponent manager was served, when the config has one;
  * the state keys the manager's encoder actually reads, so a claim about
    which channel carries the defect is shown rather than assumed;
  * how often the path fires: the realised timeout rate over agent-rounds.

Run it before and after the fix: the served value must go from the imputed
default to 0 while the env's recorded value stays at the default.

Usage (Raven, torch_geometric needed to unpickle the GNNs):
    python scripts/data_analysis/rl_manager_timeout_probe.py \\
        configs/training/rl_manager/03_2g8a_sum.yml \\
        --batch-size 32 --out plots/data_analysis/<dir>/<name>.json
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch as th
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aimanager import rl_manager  # noqa: E402
from aimanager.artificial_humans import AH_MODELS  # noqa: E402
from aimanager.manager.environment import ArtificialHumanEnv  # noqa: E402
from aimanager.manager.manager import ArtificalManager  # noqa: E402
from aimanager.manager.memory import Memory  # noqa: E402


def counts(counter):
    return {str(k): v for k, v in sorted(counter.items())}


def timeout_cells(state):
    """(contribution, valid) columns of a state dict, as flat numpy arrays."""
    c = state["contribution"].detach().cpu().reshape(-1).numpy().astype(float)
    ok = state["contribution_valid"].detach().cpu().reshape(-1).numpy().astype(bool)
    return c, ok


def resolve(basedir, rel):
    """The configured artifact, or the single `.pt` beside it.

    `configs/training/rl_manager/03_2g8a_sum.yml` names checkpoints that are
    no longer on disk (its own NOTE says as much for the opponent). Falling
    back to the one file in the same model directory keeps the probe runnable
    without editing the config, and every substitution is reported.
    """
    path = os.path.join(basedir, rel)
    if os.path.exists(path):
        return path, None
    candidates = sorted(glob.glob(os.path.join(os.path.dirname(path), "*.pt")))
    if len(candidates) != 1:
        raise FileNotFoundError(f"{path} is missing and has no unique stand-in")
    return candidates[0], f"{rel} -> {os.path.relpath(candidates[0], basedir)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--opponent",
        default=None,
        help=(
            "stand-in for `opponent_manager`; the checkpoint 03_2g8a_sum.yml "
            "names was removed and its own NOTE points new runs at "
            "artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled/"
        ),
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.opponent:
        cfg["opponent_manager"] = args.opponent

    seed = cfg["seed"]
    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    basedir = cfg["basedir"]
    device = th.device("cpu")
    substitutions = []

    def load(key):
        path, note = resolve(basedir, cfg[key])
        if note:
            substitutions.append(note)
        return AH_MODELS[cfg["artificial_humans_model"]].load(path, device=device)

    ah = load("artificial_humans")
    ahv = load("artificial_humans_valid")
    switch_model = load("switch_model") if "switch_model" in cfg else None
    opponent = load("opponent_manager") if "opponent_manager" in cfg else None

    env_args = dict(cfg["env_args"])
    rl_group_id = env_args.pop("rl_group_id", 0)
    env_args.pop("reward_formula", None)
    env_args["batch_size"] = args.batch_size
    env = ArtificialHumanEnv(
        artifical_humans=ah,
        artifical_humans_valid=ahv,
        artifical_humans_switch=switch_model,
        device=device,
        **env_args,
    )

    manager_args = cfg["manager_args"]
    manager = ArtificalManager(
        n_contributions=env.n_contributions,
        n_punishments=env.n_punishments,
        n_groups=env.n_groups,
        default_values=ah.default_values,
        device=device,
        **manager_args,
    )

    model_args = manager_args["model_args"]
    reads = [n["name"] for n in model_args["x_encoding"]]
    reads += [n["name"] for n in model_args["b_encoding"]]
    rl_manager.replay_keys = sorted(set(reads + ["punishment", "agent_group"]))

    replay_mem = Memory(
        n_episode_steps=env.n_rounds, device=device, **cfg["replay_memory_args"]
    )

    served = Counter()  # what get_action saw at a timeout cell
    recorded = Counter()  # what env.state held at the same moment
    replayed = Counter()  # what the TD update trains on
    opp_served = Counter()
    cells = {"agent_rounds": 0, "timeouts": 0}

    inner_action = manager.get_action
    # A digest of every punishment the manager chose, so the two arms can be
    # compared on the manager's own behaviour rather than only on its input.
    actions = hashlib.md5()
    action_sum = {"total": 0}

    def probed_action(state, **kw):
        c, ok = timeout_cells(state)
        cells["agent_rounds"] += len(ok)
        cells["timeouts"] += int((~ok).sum())
        for x in c[~ok]:
            served[x] += 1
        raw_c, raw_ok = timeout_cells(env.state)
        for x in raw_c[~raw_ok]:
            recorded[x] += 1
        action, q_values = inner_action(state, **kw)
        a = action.detach().cpu().numpy().astype(np.int64)
        actions.update(a.tobytes())
        action_sum["total"] += int(a.sum())
        return action, q_values

    manager.get_action = probed_action

    if opponent is not None:
        inner_predict = opponent.predict

        def probed_predict(state, **kw):
            c, ok = timeout_cells(state)
            for x in c[~ok]:
                opp_served[x] += 1
            return inner_predict(state, **kw)

        opponent.predict = probed_predict

    inner_add = replay_mem.add

    def probed_add(**kw):
        if "contribution" in kw and "contribution_valid" in kw:
            c, ok = timeout_cells(kw)
            for x in c[~ok]:
                replayed[x] += 1
        return inner_add(**kw)

    replay_mem.add = probed_add

    rl_manager.run_batch(
        manager,
        env,
        replay_mem,
        on_policy=False,
        update_step=0,
        opponent_manager=opponent,
        rl_group_id=rl_group_id,
    )

    n = cells["agent_rounds"]
    out = {
        "config": args.config,
        "batch_size": args.batch_size,
        "artifact_substitutions": substitutions,
        "manager_reads": sorted(set(reads)),
        "replay_keys": rl_manager.replay_keys,
        "agent_rounds": n,
        "timeouts": cells["timeouts"],
        "timeout_rate": cells["timeouts"] / n if n else float("nan"),
        "manager_served_at_timeouts": counts(served),
        "env_recorded_at_timeouts": counts(recorded),
        "replay_buffer_at_timeouts": counts(replayed),
        "opponent_served_at_timeouts": counts(opp_served),
        "manager_action_sum": action_sum["total"],
        "manager_action_md5": actions.hexdigest(),
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
