"""Emit the cross-check configs that re-run the fitted rules through
`simulate.py`, the path every earlier rule arm used.

The sweep harness is new: it runs the same env, the same artifacts and the
same protocol, but on the batch dimension and through its own rollout loop
rather than through `simulate.py`'s per-episode one. That is a lot of new
code between the models and the numbers, so the fitted rules and the
incumbents are also run the established way, in the established 14-pairing
shape, and the two paths are compared on the rules both can run
(`thr9_p10`, `never`, the clone). If they disagree, the sweep is wrong.

`RuleBasedManager(rule="sigmoid", ...)` shares its arithmetic with the
batched rule -- both call `sigmoid_punishment` -- so the cross-check is of
the harness, not of the formula.

Usage:
    python scripts/rule_sigmoid/sim_configs.py \
        --validation runs/validation.csv --seeds 42,43,44 \
        --out-dir configs/simulation/manager_testing
"""

import argparse
import os

import pandas as pd
import yaml

STACK = {
    "contribution_model": (
        "artifacts/artificial_humans/"
        "group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/"
        "model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
    ),
    "valid_model": "artifacts/artificial_humans/raven_script_22/model/"
    "rnn_False__dataset_full.pt",
    "switch_model": "artifacts/artificial_humans/switch_joint_exodus/model/"
    "architecture_mlp+rnn+edge__dataset_50ep_doubled.pt",
}
CLONE = "artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib"

#: The rules the cross-check carries: the two fitted optima, the incumbent,
#: the do-nothing control and the clone.
FOCALS = ("opt_pool", "opt_contribution")

HEADER = """\
# auto/rule-sigmoid-family, seed {seed}.
#
# The cross-check for the batched sweep. The fitted optima of the sigmoid
# family, the incumbent `thr9_p10` and the two controls, run through the
# path every earlier rule arm used -- `simulate.py`, one episode at a time,
# 100 episodes, `reseed_per_run`. `thr9_p10`, `never` and the clone appear in
# both this file and the sweep, so the two harnesses can be compared on rules
# they both run; the fitted rules appear so the headline is not resting on
# new code alone.
#
# Same stack, same artifacts, same 2x8 / 24-round / 100-episode protocol as
# `26_rule_inverted_targeting_s{seed}.yml`.
"""


def sigmoid_manager(row):
    return {
        "type": "rule_based",
        "rule": "sigmoid",
        "p_max": float(row["p_max"]),
        "c0": float(row["c0"]),
        "tau": float(row["tau"]),
        "gamma_ep": float(row["gamma_ep"]),
        "gamma_sw": float(row["gamma_sw"]),
    }


def build(validation, seed):
    v = validation.set_index("name")
    managers = {
        "never": {"type": "rule_based", "rule": "never"},
        "ah_punisher": {"type": "linear", "path": CLONE},
        "thr9_p10": {
            "type": "rule_based",
            "rule": "threshold",
            "threshold": 9,
            "amount": 10,
        },
    }
    for name in FOCALS:
        managers[name] = sigmoid_manager(v.loc[name])

    focals = list(FOCALS) + ["thr9_p10"]
    pairings = [
        {
            "name": "ah_punisher_vs_ah_punisher",
            "group_0": "ah_punisher",
            "group_1": "ah_punisher",
        },
        {"name": "never_vs_never", "group_0": "never", "group_1": "never"},
    ]
    for f in focals + ["never"]:
        pairings.append(
            {"name": f"{f}_vs_ah_punisher", "group_0": f, "group_1": "ah_punisher"}
        )
    for f in focals + ["ah_punisher"]:
        pairings.append({"name": f"{f}_vs_never", "group_0": f, "group_1": "never"})

    return {
        "seed": seed,
        "reseed_per_run": True,
        "artificial_humans": {
            "group_switching": {
                "contribution_model": STACK["contribution_model"],
                "valid_model": STACK["valid_model"],
                "switch_model": STACK["switch_model"],
            }
        },
        "managers": managers,
        "pairings": pairings,
        "switch_every": 4,
        "n_episode_steps": 24,
        "n_episodes": 100,
        "n_groups": 2,
        "n_agents": 8,
        "agent_groups": [0, 0, 0, 0, 1, 1, 1, 1],
        "n_contributions": 21,
        "n_punishments": 31,
        "n_rounds": 24,
        "output_dir": f"plots/simulation/27_rule_sigmoid_paired_s{seed}",
        "figure_name": f"rule_sigmoid_paired_s{seed}",
        "save_per_round": True,
        "basedir": ".",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out-dir", default="configs/simulation/manager_testing")
    args = ap.parse_args()

    validation = pd.read_csv(args.validation)
    os.makedirs(args.out_dir, exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        path = os.path.join(args.out_dir, f"27_rule_sigmoid_paired_s{seed}.yml")
        with open(path, "w") as f:
            f.write(HEADER.format(seed=seed))
            yaml.safe_dump(build(validation, seed), f, sort_keys=False)
        print(path)


if __name__ == "__main__":
    main()
