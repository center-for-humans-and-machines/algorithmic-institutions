"""Punishment conditioned on `contribution_valid`, for a trained policy.

This is the required output that cannot come from `per_round.parquet`: that
file carries punishment, common good, contribution and membership, and nothing
about validity. It has to be asked of the environment.

Two quantities, and the difference between them is the point:

  * **realised** punishment -- what the game charged and what every artificial
    human saw. Since `auto/free-punishment-fix` the env zeroes this at
    `~contribution_valid`, so it is 0 there by construction and a nonzero
    reading would mean the fix has regressed. This is the regression check.
  * **intended** action -- the level the policy's argmax actually picked,
    before the env zeroed it. The env now makes those cells a don't-care
    region rather than a free lever, so the policy may park anything there;
    what this says is whether the learned policy *targets* timed-out players,
    which is the behavioural question the review's D1 raised.

Reported for every seed, alongside the same split for the artificial punisher
so the learned policies have something to be read against.

Usage (Raven):
    python scripts/rl_two_worlds/validity_conditioned.py \\
        configs/training/rl_manager/rl_new_clones_s42.yml \\
        artifacts/manager/rl_new_clones_s42/model/rl_new_clones_s42_manager.pt \\
        --out plots/data_analysis/evaluation/rl_manager_two_worlds/validity_s42.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch as th
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# `scripts/` is not a package (adding an __init__.py there would change how
# the other script directories import), so the sibling is imported by path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aimanager.manager.manager import ArtificalManager  # noqa: E402
from launch_guards import build_env  # noqa: E402


def summarise(values, mask, label):
    """Mean, share above zero and count on one side of the validity split."""
    sel = values[mask]
    if sel.size == 0:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": int(sel.size),
        "mean": float(sel.mean()),
        "share_above_zero": float((sel > 0).mean()),
        "max": float(sel.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    device = th.device("cpu")
    seed = cfg["seed"]
    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env, opponent = build_env(cfg, args.batch_size, device)
    manager = ArtificalManager.load(
        os.path.join(cfg["basedir"], args.checkpoint), device=device
    )
    rl_group = cfg["env_args"].get("rl_group_id", 0)

    intended, realised, valid, is_rl = [], [], [], []
    opp_realised, opp_valid = [], []

    env.reset()
    for rnd in range(env.n_rounds):
        state = env.served_state()
        # greedy: the deployed policy, not the epsilon-greedy behaviour policy
        action, _ = manager.get_action(state, first=rnd == 0, greedy=True)
        opp, _ = opponent.predict(
            state, reset_rnn=rnd == 0, edge_index=env.batch_edge_index
        )
        rl_mask = (env.agent_groups.squeeze(-1) == rl_group).unsqueeze(-1)
        final = th.where(rl_mask, action, opp)

        v = state["contribution_valid"].cpu().reshape(-1).numpy().astype(bool)
        m = rl_mask.cpu().reshape(-1).numpy().astype(bool)
        intended.append(action.cpu().reshape(-1).numpy())
        valid.append(v)
        is_rl.append(m)
        opp_realised.append(opp.cpu().reshape(-1).numpy())
        opp_valid.append(v)

        env.punish(final)
        realised.append(env.state["punishment"].cpu().reshape(-1).numpy())
        env.step()

    intended = np.concatenate(intended)
    realised = np.concatenate(realised)
    valid = np.concatenate(valid)
    is_rl = np.concatenate(is_rl)
    opp_realised = np.concatenate(opp_realised)
    opp_valid = np.concatenate(opp_valid)

    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seed": seed,
        "learned_intended": {
            "valid": summarise(intended, is_rl & valid, "gave input"),
            "invalid": summarise(intended, is_rl & ~valid, "timed out"),
        },
        "learned_realised": {
            "valid": summarise(realised, is_rl & valid, "gave input"),
            "invalid": summarise(realised, is_rl & ~valid, "timed out"),
        },
        "artificial_punisher_intended": {
            "valid": summarise(opp_realised, ~is_rl & opp_valid, "gave input"),
            "invalid": summarise(opp_realised, ~is_rl & ~opp_valid, "timed out"),
        },
    }
    inv = report["learned_realised"]["invalid"]
    report["regression_check_realised_zero_at_timeout"] = (
        inv.get("n", 0) == 0 or inv.get("max") == 0.0
    )

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
