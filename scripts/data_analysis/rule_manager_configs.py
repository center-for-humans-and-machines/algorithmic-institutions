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
PAIRED_SLUG = "25_rule_vs_clone_paired"
INVERTED_SLUG = "26_rule_inverted_targeting"

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
# `reseed_per_run: true` restarts every run from `seed`, so every manager in
# this file starts its episode 0 from the same state. The streams diverge
# after that -- a manager that punishes differently moves the simulated
# players differently, and MultiManager evaluates every manager in the file
# each round, so the RNG a run consumes depends on the file's manager SET.
# Two managers are therefore strictly stream-comparable only inside one
# config; across configs the difference is an ordinary redraw, which the
# head-to-head config and the extra seeds are there to bound.
"""

HEAD_TO_HEAD_HEADER = """\
# auto/rule-based-manager-sweep, head-to-head arm, seed {seed}.
#
# The decisive comparisons in ONE config so that every manager here consumes
# exactly the same RNG per round and their difference is as close to the rule
# alone as this simulation allows: never-punish, the artificial punisher
# (the baseline to beat), the sweep's three best threshold/proportional
# rules, the human-shaped rule, and skip_invalid twins of two of them that
# never punish a player who gave no input (review finding D1). Same stack,
# same protocol, same seed discipline as the sweep shards.
"""

HEAD_TO_HEAD = [
    "never",
    "ah_punisher",
    "prop10",
    "thr9_p10",
    "thr9_p5",
    "human_severity",
]
SKIP_INVALID_TWINS = ["prop10", "thr9_p5"]

# ---------------------------------------------------------------------- #
# paired arm: a rule and a RIVAL in the same world, one group each
# ---------------------------------------------------------------------- #
PAIRED_MANAGERS = [
    "never",
    "ah_punisher",
    "prop10",
    "thr9_p10",
    "thr9_p5",
    "human_severity",
]
PAIRED_FOCALS = ["prop10", "thr9_p10", "thr9_p5", "human_severity"]

# ---------------------------------------------------------------------- #
# inverted-targeting arm: punishing the WRONG people, in the paired world
# ---------------------------------------------------------------------- #
# Kept in its own dict rather than folded into `rule_managers()` so the
# sweep's shard layout and the paired arm's manager set cannot drift.
#
# `thr9_p10` punishes 10 on c <= 9: ten of the twenty-one levels. Two
# mirrors are run, because the two natural senses of "mirror" disagree here
# and the disagreement is itself a measurement (see the log, section 3.1):
#
# * `inv_thr11_p10` -- the exact reflection of `thr9_p10` under c -> 20 - c.
#   Matched on LEVELS: ten levels each, the same amount, both sparing the
#   midpoint c = 10. On the untreated contribution distribution
#   (`never_vs_never`, three seeds) it fires on 0.342 of cells against
#   `thr9_p10`'s 0.565, because contributions in this world are massed low
#   -- so it UNDER-spends, and a loss under it is a conservative reading.
# * `inv_thr7_p10` -- matched on SPEND instead: P(c >= 7) = 0.567 against
#   P(c <= 9) = 0.565, a gap of 0.15 percentage points, and the same match
#   holds on valid-only cells (0.558 against 0.557). It costs the level
#   match (fourteen levels, not ten) to buy the intensity match.
#
# Running both brackets the mirror choice: if the two agree, the reading
# does not depend on which sense of "mirror" is used.
#
# TWO MORE rules test a separate hypothesis the broad mirrors cannot: that
# punishing NEARLY-FULL contributors is a valid strategy, because they are
# close to the ceiling and so cheap to push the rest of the way up.
# `inv_thr11_p10` does not test it -- firing from 11 upward spans the whole
# withdrawal zone, mixing the idea with ordinary indiscriminate
# over-punishment. `band16_*` fires ONLY on the near-ceiling band, c >= 16,
# and leaves everyone below untouched.
#
# A narrow band in this distribution cannot both fire gently and spend as
# much as a broad rule -- only 0.186 of cells sit at c >= 16 -- so the band
# is run at two intensities and the reader is given both realised means:
#
# * `band16_p10` -- untreated spend 1.86, matched to the CLONE's realised
#   1.9-2.0. The modest-intensity near-ceiling rule.
# * `band16_p20` -- untreated spend 3.72, matched to `thr9_p10`'s REALISED
#   3.71 in the parent's paired arm and to `inv_thr11_p10`'s 3.42. It buys
#   the spend match by punishing each hit twice as hard.
#
# Together they separate the BAND from the INTENSITY inside the
# near-ceiling family, which is what decides whether amount dominates
# direction.
INVERTED_RULES = {
    "inv_thr11_p10": {
        "type": "rule_based",
        "rule": "inv_threshold",
        "threshold": 11,
        "amount": 10,
    },
    "inv_thr7_p10": {
        "type": "rule_based",
        "rule": "inv_threshold",
        "threshold": 7,
        "amount": 10,
    },
    "band16_p10": {
        "type": "rule_based",
        "rule": "inv_threshold",
        "threshold": 16,
        "amount": 10,
    },
    "band16_p20": {
        "type": "rule_based",
        "rule": "inv_threshold",
        "threshold": 16,
        "amount": 20,
    },
}
INVERTED_BORROWED = ["never", "ah_punisher", "thr9_p10"]
INVERTED_FOCALS = [
    "inv_thr11_p10",
    "inv_thr7_p10",
    "band16_p10",
    "band16_p20",
    "thr9_p10",
]

INVERTED_HEADER = """\
# auto/rule-inverted-targeting, seed {seed}.
#
# The learned RL managers came out INVERTED: they punish full contributors
# hardest and leave free-riders alone. An intervention probe established
# that the contributor model's targeting signal is real but weak, and that
# at the ceiling -- where two of three learned seeds aim -- the response is
# close to inert. That is a hypothesis about CONSEQUENCES: it says punishing
# the wrong people may be nearly free. This arm tests it by simulating an
# inverted manager rather than arguing from the probe.
#
# Run in the COMPETING setting, not self-play, because the parent branch
# established that self-play rankings do not survive competition (`prop10`
# went from +12.5 per seat in self-play to -15.6 against a live rival).
#
# Five focal rules against the same two rivals the paired arm used:
#   thr9_p10       punish 10 on c <= 9    (the correctly-targeted rule)
#   inv_thr11_p10  punish 10 on c >= 11   (inverted, level-matched mirror)
#   inv_thr7_p10   punish 10 on c >= 7    (inverted, spend-matched mirror)
#   band16_p10     punish 10 on c >= 16   (near-ceiling band, clone intensity)
#   band16_p20     punish 20 on c >= 16   (near-ceiling band, matched spend)
#
# The band rules test a hypothesis the broad mirrors cannot: that punishing
# nearly-full contributors is a valid strategy because they are cheap to
# push the rest of the way to the ceiling. Firing from 11 upward spans the
# whole withdrawal zone and mixes that idea with indiscriminate
# over-punishment; firing only from 16 upward does not.
#
# `thr9_p10` and `never` are re-run HERE rather than quoted from the parent,
# because MultiManager evaluates every manager in a file each round, so the
# RNG a run consumes depends on the file's manager SET (parent log, note 4).
# One file, one manager set, so all five focals are stream-comparable.
#
# Same stack, same artifacts, same 2x8 / 24-round / 100-episode protocol and
# the same `reseed_per_run` seed discipline as the parent's paired arm.
"""


def inverted_managers(basedir="."):
    """The inverted arm's manager set: two mirrors plus the three borrowed."""
    all_m = rule_managers(basedir)
    managers = {k: all_m[k] for k in INVERTED_BORROWED}
    managers.update(INVERTED_RULES)
    return managers


def inverted_pairings():
    """(focal, rival) seats, controls first, in report order."""
    pairs = [("ah_punisher", "ah_punisher"), ("never", "never")]
    pairs += [(f, "ah_punisher") for f in INVERTED_FOCALS] + [("never", "ah_punisher")]
    pairs += [(f, "never") for f in INVERTED_FOCALS] + [("ah_punisher", "never")]
    return [{"name": f"{f}_vs_{r}", "group_0": f, "group_1": r} for f, r in pairs]


PAIRED_HEADER = """\
# auto/rule-vs-clone-paired, seed {seed}.
#
# The sweep ran every manager in SELF-PLAY: MultiManager separates managers
# along the batch dimension, so each manager governed its own parallel
# population and none of them ever faced another. This arm puts a rule and a
# RIVAL in the SAME world, one group each, so a manager can win or lose
# members to the other seat -- the game the RL manager is actually trained in.
#
# Two rival families, because they are different games. Against the clone
# (`ah_punisher`, our clone of a human manager) both seats punish and members
# choose between two disciplined groups. Against `never` one seat is a refuge
# with no punishment at all, so a punishing rule asks its members to accept a
# cost they could avoid by moving -- the harder test of whether punishing
# survives when leaving is easy.
#
# `{{focal}}_vs_{{rival}}`: group_0 carries the focal manager, group_1 the
# rival. `ah_punisher_vs_ah_punisher` and `never_vs_never` are the symmetric
# controls that say whether an asymmetry is the rule or the seat;
# `never_vs_ah_punisher` and `ah_punisher_vs_never` are seat swaps of each
# other and connect the two families.
#
# Same stack, same artifacts, same 2x8 / 24-round / 100-episode protocol and
# the same `reseed_per_run` seed discipline as the sweep's head-to-head arm,
# and ONE manager set for all pairings so every run draws from the same
# manager population (sweep log, note 4).
"""


def paired_pairings():
    """(focal, rival) seats, controls first, in report order."""
    pairs = [("ah_punisher", "ah_punisher"), ("never", "never")]
    pairs += [(f, "ah_punisher") for f in PAIRED_FOCALS] + [("never", "ah_punisher")]
    pairs += [(f, "never") for f in PAIRED_FOCALS] + [("ah_punisher", "never")]
    return [{"name": f"{f}_vs_{r}", "group_0": f, "group_1": r} for f, r in pairs]


def write_paired_config(
    path,
    seed,
    managers,
    out_dir_suffix,
    pairings=None,
    header=None,
    slug=PAIRED_SLUG,
    figure_stem="rule_vs_clone_paired",
):
    config = {
        "seed": seed,
        "reseed_per_run": True,
        "artificial_humans": {"group_switching": dict(STACK)},
        "managers": managers,
        "pairings": paired_pairings() if pairings is None else pairings,
        "switch_every": 4,
        "n_episode_steps": 24,
        "n_episodes": 100,
        "n_groups": 2,
        "n_agents": 8,
        "agent_groups": [0, 0, 0, 0, 1, 1, 1, 1],
        "n_contributions": 21,
        "n_punishments": 31,
        "n_rounds": 24,
        "output_dir": f"{OUTPUT_ROOT}/{slug}_{out_dir_suffix}",
        "figure_name": f"{figure_stem}_{out_dir_suffix}",
        "save_per_round": True,
        "basedir": ".",
    }
    with open(path, "w") as f:
        f.write(PAIRED_HEADER.format(seed=seed) if header is None else header)
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {path}  ({len(config['pairings'])} pairings)")


def write_config(path, shard, n_shards, seed, managers, out_dir_suffix, header=None):
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
        if header is None:
            header = HEADER.format(shard=shard, n_shards=n_shards, seed=seed)
        f.write(header)
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {path}  ({len(managers)} managers)")


def head_to_head_managers(basedir="."):
    all_m = rule_managers(basedir)
    managers = {k: all_m[k] for k in HEAD_TO_HEAD}
    for k in SKIP_INVALID_TWINS:
        managers[f"{k}_si"] = {**all_m[k], "skip_invalid": True}
    return managers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--per-shard", type=int, default=7)
    ap.add_argument("--head-to-head", action="store_true")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--inverted", action="store_true")
    ap.add_argument("--basedir", default=".")
    args = ap.parse_args()

    if args.inverted:
        managers = inverted_managers(args.basedir)
        pairings = inverted_pairings()
        for seed in args.seeds:
            suffix = f"s{seed}"
            write_paired_config(
                os.path.join(args.basedir, CONFIG_DIR, f"{INVERTED_SLUG}_{suffix}.yml"),
                seed=seed,
                managers=managers,
                out_dir_suffix=suffix,
                pairings=pairings,
                header=INVERTED_HEADER.format(seed=seed),
                slug=INVERTED_SLUG,
                figure_stem="rule_inverted_targeting",
            )
        return

    if args.paired:
        all_m = rule_managers(args.basedir)
        managers = {k: all_m[k] for k in PAIRED_MANAGERS}
        for seed in args.seeds:
            suffix = f"s{seed}"
            write_paired_config(
                os.path.join(args.basedir, CONFIG_DIR, f"{PAIRED_SLUG}_{suffix}.yml"),
                seed=seed,
                managers=managers,
                out_dir_suffix=suffix,
            )
        return

    if args.head_to_head:
        managers = head_to_head_managers(args.basedir)
        for seed in args.seeds:
            suffix = f"h2h_s{seed}"
            write_config(
                os.path.join(args.basedir, CONFIG_DIR, f"{SLUG}_{suffix}.yml"),
                shard=1,
                n_shards=1,
                seed=seed,
                managers=managers,
                out_dir_suffix=suffix,
                header=HEAD_TO_HEAD_HEADER.format(seed=seed),
            )
        return

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
