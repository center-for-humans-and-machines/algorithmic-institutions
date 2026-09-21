"""Emit the rule-based-manager sweep's simulation configs.

One family of simple, interpretable punishment rules is run against the
frontier artificial-human stack, alongside the artificial punisher (our
clone of a human manager) in the identical setup. The rules are defined
here, in one place, so the sweep's space can be read off a single file;
the YAML this writes is committed and is what the cluster runs.

The two human-shaped rules are read off the human data
(`experiments/2group_8agent_50ep.csv`, single copy per game, rows where
both the player and the manager gave an input), so their tables are
derived, not invented -- this script bakes them into the YAML.

Usage:
    python scripts/data_analysis/rule_manager_configs.py
    python scripts/data_analysis/rule_manager_configs.py --seeds 42 43 44
"""

import argparse
import os

import pandas as pd
import yaml

HUMAN_DATA = "experiments/2group_8agent_50ep.csv"
CONFIG_DIR = "configs/simulation/manager_testing"
OUTPUT_ROOT = "plots/simulation"
SLUG = "24_rule_managers"

STACK = {
    "contribution_model": (
        "artifacts/artificial_humans/"
        "group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/"
        "model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
    ),
    "valid_model": (
        "artifacts/artificial_humans/raven_script_22/model/rnn_False__dataset_full.pt"
    ),
    "switch_model": (
        "artifacts/artificial_humans/switch_joint_exodus/"
        "model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt"
    ),
}
PUNISHER = "artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib"

THRESHOLDS = [4, 9, 14, 19]
AMOUNTS = [2, 5, 10]
RATES = [0.1, 0.25, 0.5, 1.0]


def human_tables(basedir="."):
    """(E[p|c], P(p>0|c), E[p|p>0,c]) over c = 0..20, human single copy."""
    df = pd.read_csv(os.path.join(basedir, HUMAN_DATA))
    keep = df.groupby("pair_id")["episode_id"].transform("min")
    df = df[df["episode_id"] == keep]
    df = df[(df["player_no_input"] == 0) & (df["manager_no_input"] == 0)]
    by_c = df.groupby(df["contribution"].astype(int))["punishment"]
    mean = by_c.mean()
    rate = by_c.apply(lambda s: (s > 0).mean())
    sev = by_c.apply(lambda s: s[s > 0].mean())
    grid = range(21)
    return (
        [float(round(mean.get(c, 0.0))) for c in grid],
        [round(float(rate.get(c, 0.0)), 4) for c in grid],
        [float(round(sev.get(c, 0.0))) for c in grid] if len(sev) else [0.0] * 21,
    )


def rule_managers(basedir="."):
    """name -> manager config dict, in the order they are reported."""
    mean_tab, rate_tab, sev_tab = human_tables(basedir)
    managers = {"never": {"type": "rule_based", "rule": "never"}}
    for t in THRESHOLDS:
        for a in AMOUNTS:
            managers[f"thr{t}_p{a}"] = {
                "type": "rule_based",
                "rule": "threshold",
                "threshold": t,
                "amount": a,
            }
    for r in RATES:
        managers[f"prop{str(r).replace('.', '')}"] = {
            "type": "rule_based",
            "rule": "proportional",
            "rate": r,
        }
    managers["human_mean"] = {
        "type": "rule_based",
        "rule": "table",
        "table": mean_tab,
    }
    managers["human_severity"] = {
        "type": "rule_based",
        "rule": "severity_table",
        "table": sev_tab,
        "prob_table": rate_tab,
    }
    managers["ah_punisher"] = {"type": "linear", "path": PUNISHER}
    return managers


HEADER = """\
# auto/rule-based-manager-sweep, shard {shard} of {n_shards}, seed {seed}.
#
# Does punishing pay at all against the corrected simulated players, and can
# any simple rule beat the artificial punisher? Each manager below plays
# SELF-PLAY (both sides of the pairing), against the frontier stack
# (contribution trunk with per-group virtual node + direct stimulus skip +
# stamped herding copula; joint-exodus switch model; validity model
# unchanged) -- byte-identical artifacts to
# 23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_simtimeout,
# and the same 2x8 agents / 24 rounds / 100 episodes protocol.
#
# `reseed_per_run: true` restarts every run from `seed`, so the difference
# between two managers here is the manager and not the position of its run
# in the file. `ah_punisher` is the baseline to beat and reproduces the
# frontier run above exactly.
"""


def write_config(path, shard, n_shards, seed, managers, out_dir_suffix):
    pairings = [
        {"name": f"{k}_self", "group_0": k, "group_1": k} for k in managers.keys()
    ]
    config = {
        "seed": seed,
        "reseed_per_run": True,
        "artificial_humans": {"group_switching": dict(STACK)},
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
        "output_dir": f"{OUTPUT_ROOT}/{SLUG}_{out_dir_suffix}",
        "figure_name": f"rule_managers_{out_dir_suffix}",
        "save_per_round": True,
        "basedir": ".",
    }
    with open(path, "w") as f:
        f.write(HEADER.format(shard=shard, n_shards=n_shards, seed=seed))
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {path}  ({len(managers)} managers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--per-shard", type=int, default=7)
    ap.add_argument("--basedir", default=".")
    args = ap.parse_args()

    managers = rule_managers(args.basedir)
    names = list(managers)
    shards = [
        names[i : i + args.per_shard] for i in range(0, len(names), args.per_shard)
    ]
    for seed in args.seeds:
        for i, shard_names in enumerate(shards):
            suffix = f"s{seed}_{chr(ord('a') + i)}"
            write_config(
                os.path.join(args.basedir, CONFIG_DIR, f"{SLUG}_{suffix}.yml"),
                shard=i + 1,
                n_shards=len(shards),
                seed=seed,
                managers={k: managers[k] for k in shard_names},
                out_dir_suffix=suffix,
            )


if __name__ == "__main__":
    main()
