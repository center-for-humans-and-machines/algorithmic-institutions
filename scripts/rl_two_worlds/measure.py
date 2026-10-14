"""The measurement plan of notes/autoresearch_log/rl-manager-two-worlds.md,
turned into tables.

Reads the cross-evaluation simulation's `per_round.parquet` and the human
reference through the evaluation suite's own canonical frame, so "punished 0"
and "the manager gave no input" stay distinct -- the human file marks a manager
timeout as a NaN punishment and pooling the two would quietly deflate every
human punishment statistic by the 4.2% of rows where no input was given.

Writes to plots/data_analysis/evaluation/rl_manager_two_worlds/:
  does_it_punish.csv   -- rate, mean, mean-given-positive, per manager
  policy_shape.csv     -- mean punishment per RPA contribution bin
  common_good.csv      -- common good and its spread, per manager
  seed_spread.md       -- the three learned seeds against each other

Usage:
    python scripts/rl_two_worlds/measure.py \\
        plots/simulation/24_rl_new_clones_cross_eval/per_round.parquet
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.evaluation_suite.convert import (  # noqa: E402
    HUMAN_DATA_FILE,
    load_human,
    load_sim,
)
from aimanager.evaluation_suite.metrics import (  # noqa: E402
    RPA_LABELS,
    ResponseMetrics,
)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/evaluation/rl_manager_two_worlds")
SEED_RUNS = ("rl_s42_vs_punisher", "rl_s43_vs_punisher", "rl_s44_vs_punisher")


def punish_stats(df, label):
    """Rate, mean and mean-given-positive over the rows where the manager
    actually gave an input. `punishment` is NaN exactly where it did not."""
    p = df["punishment"].dropna()
    positive = p[p > 0]
    return {
        "manager": label,
        "n_agent_rounds": int(len(p)),
        "punish_rate": float((p > 0).mean()),
        "mean_punishment": float(p.mean()),
        "mean_given_positive": float(positive.mean()) if len(positive) else 0.0,
    }


def policy_shape(df, label):
    """Mean punishment per contribution bin -- the manager's policy, on the
    evaluation suite's own RPA bins so it is the same axis the suite plots."""
    s = ResponseMetrics().rpa(df).groupby(level=0).mean().reindex(RPA_LABELS)
    return s.rename(label)


def group0_only(df):
    """The cross-evaluation puts the manager under test in group 0; group 1 is
    always the artificial punisher, so its rows would dilute every statistic."""
    return df[df["group_id"] == 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("per_round")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    human = load_human(os.path.join(ROOT, HUMAN_DATA_FILE))
    sims = load_sim(args.per_round)
    os.makedirs(args.out, exist_ok=True)

    rows = [punish_stats(human, "human managers")]
    shapes = [policy_shape(human, "human managers")]
    for name, df in sorted(sims.items()):
        g0 = group0_only(df)
        rows.append(punish_stats(g0, name))
        shapes.append(policy_shape(g0, name))

    punish = pd.DataFrame(rows).set_index("manager")
    punish.to_csv(os.path.join(args.out, "does_it_punish.csv"))
    shape = pd.concat(shapes, axis=1)
    shape.index.name = "contribution_bin"
    shape.to_csv(os.path.join(args.out, "policy_shape.csv"))

    cg = pd.DataFrame(
        [
            {
                "manager": name,
                "common_good_mean": (
                    float(group0_only(df)["common_good"].mean())
                    if "common_good" in df
                    else float("nan")
                ),
            }
            for name, df in sorted(sims.items())
        ]
    ).set_index("manager")
    cg.to_csv(os.path.join(args.out, "common_good.csv"))

    # The seed spread is the thing being measured, not a nuisance: report the
    # three learned runs against each other and against the gap to the
    # baseline, so "do the seeds agree" is answerable at a glance.
    present = [r for r in SEED_RUNS if r in sims]
    lines = ["# Seed spread", ""]
    if len(present) < 2:
        lines.append(f"Only {len(present)} of the three seeds present.")
    else:
        sub = punish.loc[present]
        base = "lin_punisher_self"
        lines.append(
            "| statistic | " + " | ".join(present) + " | spread | "
            "baseline (artificial punisher) |"
        )
        lines.append("|---" * (len(present) + 3) + "|")
        for col in ("punish_rate", "mean_punishment", "mean_given_positive"):
            vals = [sub.loc[r, col] for r in present]
            spread = max(vals) - min(vals)
            b = punish.loc[base, col] if base in punish.index else float("nan")
            lines.append(
                f"| {col} | "
                + " | ".join(f"{v:.4f}" for v in vals)
                + f" | {spread:.4f} | {b:.4f} |"
            )
        lines += [
            "",
            "Read the spread against the gap to the baseline: three seeds far "
            "apart relative to that gap means more seeds, not a mean over "
            "three.",
        ]
    with open(os.path.join(args.out, "seed_spread.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(punish.to_string())
    print()
    print(shape.to_string())


if __name__ == "__main__":
    main()
