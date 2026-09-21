"""The two pre-launch guards for the new-clones RL runs.

Both are asked of the **real** training environment built from the real config
-- the same models, the same env, the same opponent -- because both failures
they look for are silent: a run trained on the old reward, or with the free
punishment lever still open, looks entirely normal until somebody checks.

GUARD 1 -- the reward really is the common pool.
    For every group and round, `env.reward` must equal 1.6 * sum(contributions)
    - sum(punishments) for that group. Checked three ways so the answer does
      not rest on this file reimplementing the env:
      a. against the pool recomputed here from the env's own state tensors;
      b. against `env.common_good * n_valid` -- `common_good` is the per-capita
         pool and is produced by a different method (`update_common_good`), so
         agreement is an independent confirmation rather than a tautology;
      c. against the same rollout under `reward_mode='sum'`, which must NOT
         match -- if it does, the mode never changed and guard 1 is vacuous.

GUARD 2 -- the free punishment lever is closed.
    A punishment aimed at a player who gave no input costs the manager nothing
    (the accounting zeroes it) but used to still reach every artificial human
    through `punishment` / `prev_punishment`. Here the manager is forced to
    punish the maximum on every cell -- the strongest possible probe -- and the
    guard reports every punishment value the contribution, validity and switch
    models were actually served at a timed-out cell. All of them must be 0.

Usage (Raven; torch_geometric is needed to unpickle the GNNs):
    python scripts/rl_two_worlds/launch_guards.py \\
        configs/training/rl_manager/rl_new_clones_s42.yml \\
        --batch-size 64 --out plots/data_analysis/<dir>/guards.json
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch as th
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aimanager.artificial_humans import AH_MODELS  # noqa: E402
from aimanager.manager.environment import ArtificialHumanEnv  # noqa: E402
from aimanager.manager.linear_opponent import load_opponent  # noqa: E402

POOL_RATE = 1.6


def build_env(cfg, batch_size, device, reward_mode=None):
    basedir = cfg["basedir"]
    kind = AH_MODELS[cfg["artificial_humans_model"]]

    def load(key):
        return kind.load(os.path.join(basedir, cfg[key]), device=device)

    env_args = dict(cfg["env_args"])
    env_args.pop("rl_group_id", 0)
    env_args.pop("reward_formula", None)
    env_args["batch_size"] = batch_size
    if reward_mode is not None:
        env_args["reward_mode"] = reward_mode
    env = ArtificialHumanEnv(
        artifical_humans=load("artificial_humans"),
        artifical_humans_valid=load("artificial_humans_valid"),
        artifical_humans_switch=(
            load("switch_model") if "switch_model" in cfg else None
        ),
        device=device,
        **env_args,
    )
    opponent = None
    if "opponent_manager" in cfg:
        opponent = load_opponent(
            os.path.join(basedir, cfg["opponent_manager"]),
            n_groups=env_args.get("n_groups", 1),
            device=device,
        ).to(device)
    return env, opponent


def pool_from_state(env):
    """1.6 * sum(c) - sum(p) per group, from the env's own state tensors.

    Invalid cells are zeroed first: a player who gave no input contributed
    nothing and, once the lever is closed, was punished nothing. That zeroing
    is what makes the identity hold on the human data.
    """
    valid = env.state["contribution_valid"]
    c = th.where(valid, env.state["contribution"], th.zeros_like(valid.long()))
    p = th.where(valid, env.state["punishment"], th.zeros_like(valid.long()))
    mask = env.agent_group_mask
    sum_c = (c.to(th.float).unsqueeze(-2) * mask).sum(dim=1)
    sum_p = (p.to(th.float).unsqueeze(-2) * mask).sum(dim=1)
    return POOL_RATE * sum_c - sum_p


def raw_pool_from_state(env):
    """The same without zeroing invalid cells. Equal to `pool_from_state` only
    if nothing was ever charged at a timed-out cell -- a second read on
    guard 2, from the accounting side rather than the serving side."""
    mask = env.agent_group_mask
    c = env.state["contribution"].to(th.float).unsqueeze(-2) * mask
    p = env.state["punishment"].to(th.float).unsqueeze(-2) * mask
    return POOL_RATE * c.sum(dim=1) - p.sum(dim=1)


def run_rollout(env, opponent, n_rounds, watch_punishment=False, seed=0):
    """Drive the env with a maximum-punishment policy and collect the guards.

    Maximum punishment everywhere is deliberate: it is the strongest possible
    test of guard 2 and it keeps the reward well away from 0, so guard 1 is not
    satisfied trivially.
    """
    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    served_at_timeout = Counter()  # punishment values the AHs saw there
    served_prev_at_timeout = Counter()
    watched = {"calls": 0, "timeout_cells": 0, "agent_cells": 0}

    if watch_punishment:
        for model in (
            env.artifical_humans,
            env.artifical_humans_valid,
            env.artifical_humans_switch,
        ):
            if model is None:
                continue
            inner = model.predict

            def probed(state, _inner=inner, **kw):
                valid = state["contribution_valid"].cpu().reshape(-1).numpy()
                ok = valid.astype(bool)
                watched["calls"] += 1
                watched["agent_cells"] += len(ok)
                watched["timeout_cells"] += int((~ok).sum())
                for key, sink in (
                    ("punishment", served_at_timeout),
                    ("prev_punishment", served_prev_at_timeout),
                ):
                    if key not in state:
                        continue
                    v = state[key].detach().cpu().reshape(-1).numpy()
                    for x in v[~ok]:
                        sink[float(x)] += 1
                return _inner(state, **kw)

            model.predict = probed

    env.reset()
    rows = []
    for rnd in range(n_rounds):
        state = env.served_state()
        action = th.full_like(env.state["punishment"], env.n_punishments - 1)
        if opponent is not None:
            opp, _ = opponent.predict(
                state, reset_rnn=rnd == 0, edge_index=env.batch_edge_index
            )
            rl_mask = (env.agent_groups.squeeze(-1) == 0).unsqueeze(-1)
            action = th.where(rl_mask, action, opp)
        env.punish(action)

        n_valid = env.count_valid_per_group(env.state["contribution_valid"])
        cg_per_agent = env.state["common_good"]
        cg_per_group = (cg_per_agent.unsqueeze(-2) * env.agent_group_mask).sum(
            dim=1
        ) / env.count_members_per_group().clamp(min=1)
        rows.append(
            {
                "round": rnd,
                "reward": env.reward.detach().clone(),
                "pool": pool_from_state(env),
                "raw_pool": raw_pool_from_state(env),
                "cg_times_nvalid": cg_per_group * n_valid,
            }
        )
        _, _, done = env.step()
        if done:
            break
    return rows, served_at_timeout, served_prev_at_timeout, watched


def residuals(rows, key):
    d = th.cat([(r["reward"] - r[key]).abs().reshape(-1) for r in rows])
    return {"max": float(d.max()), "mean": float(d.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    device = th.device("cpu")
    report = {"config": args.config, "reward_mode": cfg["env_args"]["reward_mode"]}

    # ---- guard 1 + guard 2, under the configured (common_pool) mode ------- #
    env, opponent = build_env(cfg, args.batch_size, device)
    rows, served, served_prev, watched = run_rollout(
        env, opponent, args.rounds, watch_punishment=True, seed=cfg["seed"]
    )
    report["guard1"] = {
        "vs_recomputed_pool": residuals(rows, "pool"),
        "vs_raw_pool_no_zeroing": residuals(rows, "raw_pool"),
        "vs_common_good_times_n_valid": residuals(rows, "cg_times_nvalid"),
        "mean_reward": float(th.cat([r["reward"].reshape(-1) for r in rows]).mean()),
    }
    report["guard2"] = {
        "punishment_values_served_at_timeout": {
            str(k): v for k, v in sorted(served.items())
        },
        "prev_punishment_values_served_at_timeout": {
            str(k): v for k, v in sorted(served_prev.items())
        },
        "timeout_cells_observed": watched["timeout_cells"],
        "agent_cells_observed": watched["agent_cells"],
        "timeout_rate": (
            watched["timeout_cells"] / watched["agent_cells"]
            if watched["agent_cells"]
            else None
        ),
    }

    # ---- guard 1c: the same rollout under `sum` must NOT match the pool --- #
    env_sum, opp_sum = build_env(cfg, args.batch_size, device, reward_mode="sum")
    rows_sum, _, _, _ = run_rollout(
        env_sum, opp_sum, args.rounds, watch_punishment=False, seed=cfg["seed"]
    )
    report["guard1c_sum_contrast"] = {
        "vs_recomputed_pool": residuals(rows_sum, "pool"),
        "mean_reward": float(
            th.cat([r["reward"].reshape(-1) for r in rows_sum]).mean()
        ),
    }

    served_nonzero = {k: v for k, v in served.items() if k != 0.0}
    prev_nonzero = {k: v for k, v in served_prev.items() if k != 0.0}
    report["verdict"] = {
        "guard1_reward_is_common_pool": (
            report["guard1"]["vs_recomputed_pool"]["max"] < 1e-3
            and report["guard1"]["vs_common_good_times_n_valid"]["max"] < 1e-3
        ),
        "guard1c_sum_differs": (
            report["guard1c_sum_contrast"]["vs_recomputed_pool"]["max"] > 1.0
        ),
        "guard2_no_punishment_at_timeout": (not served_nonzero and not prev_nonzero),
        "guard2_saw_timeouts": watched["timeout_cells"] > 0,
    }
    report["verdict"]["ALL_PASS"] = all(report["verdict"].values())

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0 if report["verdict"]["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
