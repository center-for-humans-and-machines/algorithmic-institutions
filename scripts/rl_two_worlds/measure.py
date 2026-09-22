"""The measurement plan of notes/autoresearch_log/rl-manager-two-worlds.md,
turned into tables.

The manager under test always sits in group 0 and the artificial punisher in
group 1, so every statistic is group 0's: group 1's rows would dilute it with
the opponent's behaviour.

Two frames, deliberately:

  * the evaluation suite's canonical frame (`convert.load_sim` /
    `load_human`) for anything compared against the humans, because it is what
    keeps "punished 0" and "the manager gave no input" apart -- the human file
    marks a manager timeout as a NaN punishment and pooling the two would
    deflate every human punishment statistic by the 4.2% of rows with no
    manager input;
  * the raw `per_round.parquet` for common good and for the validity split,
    because the canonical frame drops `common_good` on purpose (its scale
    differs between sources) and the human file has no simulated counterpart
    for `contribution_valid`.

Writes to plots/data_analysis/evaluation/rl_manager_two_worlds/:
  does_it_punish.csv   -- rate, mean, mean-given-positive, per manager
  policy_shape.csv     -- mean punishment per RPA contribution bin
  policy_shape_n.csv   -- rows behind each of those means
  outcomes.csv         -- common good, contribution, group size, per manager
  validity_split.csv   -- realised punishment by contribution_valid
  seed_spread.md       -- the three seeds against each other and the baseline

Usage:
    python scripts/rl_two_worlds/measure.py \\
        plots/simulation/24_rl_new_clones_cross_eval/per_round.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.evaluation_suite.convert import (  # noqa: E402
    HUMAN_DATA_FILE,
    load_human,
    load_sim,
)
from aimanager.evaluation_suite.metrics import (  # noqa: E402
    RPA_EDGES,
    RPA_LABELS,
    ResponseMetrics,
)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/evaluation/rl_manager_two_worlds")
SEEDS = ("rl_s42", "rl_s43", "rl_s44")
BASELINE = "lin_punisher"


def short(name):
    """`ah group_switching managed by rl_s42_vs_punisher` -> `rl_s42`."""
    for a, b in (
        ("ah group_switching managed by ", ""),
        ("_vs_punisher", ""),
        ("_self", ""),
    ):
        name = name.replace(a, b)
    return name


def punish_stats(df, label):
    p = df["punishment"].dropna()
    pos = p[p > 0]
    return {
        "manager": label,
        "n_agent_rounds": int(len(p)),
        "punish_rate": float((p > 0).mean()),
        "mean_punishment": float(p.mean()),
        "mean_given_positive": float(pos.mean()) if len(pos) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("per_round")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    human = load_human(os.path.join(ROOT, HUMAN_DATA_FILE))
    sims = {short(k): v for k, v in load_sim(args.per_round).items()}
    raw = pd.read_parquet(args.per_round)
    raw["run"] = raw["run"].map(short)
    g0 = raw[raw["group_id"] == 0]

    # ---- 1. does it punish -------------------------------------------- #
    R = ResponseMetrics()
    rows, shapes, counts = [punish_stats(human, "human managers")], {}, {}
    shapes["human managers"] = R.rpa(human).groupby(level=0).mean().reindex(RPA_LABELS)
    hb = pd.cut(
        human.dropna(subset=["punishment", "contribution"])["contribution"],
        RPA_EDGES,
        labels=RPA_LABELS,
    ).astype(str)
    counts["human managers"] = hb.value_counts().reindex(RPA_LABELS)
    for name, df in sorted(sims.items()):
        d = df[df["group_id"] == 0]
        stats = punish_stats(d, name)
        # The suite's own RPA discrepancy: the human-frequency-weighted mean
        # of the per-bin 1-Wasserstein distances between this manager's
        # punishment distribution and the humans'. Raw, not noise-ceiling
        # normalised -- comparable between the rows of this table, not with a
        # scores.csv figure.
        stats["rpa_distance_vs_human"] = float(R.d("RPA", human, d))
        rows.append(stats)
        shapes[name] = R.rpa(d).groupby(level=0).mean().reindex(RPA_LABELS)
        b = pd.cut(
            d.dropna(subset=["punishment", "contribution"])["contribution"],
            RPA_EDGES,
            labels=RPA_LABELS,
        ).astype(str)
        counts[name] = b.value_counts().reindex(RPA_LABELS)
    punish = pd.DataFrame(rows).set_index("manager")
    punish.to_csv(os.path.join(args.out, "does_it_punish.csv"))
    shape = pd.DataFrame(shapes)
    shape.index.name = "contribution_bin"
    shape.to_csv(os.path.join(args.out, "policy_shape.csv"))
    cnt = pd.DataFrame(counts)
    cnt.index.name = "contribution_bin"
    cnt.to_csv(os.path.join(args.out, "policy_shape_n.csv"))

    # ---- 2. outcomes: common good, per episode so the CI is honest ----- #
    per_round = (
        g0.groupby(["run", "episode", "round_number"])
        .agg(
            cg=("common_good", "mean"),
            contr=("contribution", "mean"),
            p=("punishment", "mean"),
            size=("participant_code", "size"),
        )
        .reset_index()
    )
    ep = per_round.groupby(["run", "episode"]).mean(numeric_only=True)
    out = ep.groupby("run").agg(
        common_good=("cg", "mean"),
        common_good_sd=("cg", "std"),
        contribution=("contr", "mean"),
        mean_punishment=("p", "mean"),
        group_size=("size", "mean"),
    )
    n_ep = ep.groupby("run")["cg"].count()
    out["common_good_ci95"] = 1.96 * out["common_good_sd"] / np.sqrt(n_ep)
    out = out.sort_values("common_good", ascending=False)
    out.to_csv(os.path.join(args.out, "outcomes.csv"))

    # ---- 3. the validity split, from the raw frame --------------------- #
    vs = (
        g0.groupby(["run", "contribution_valid"])["punishment"]
        .agg(["size", "mean", lambda s: float((s > 0).mean())])
        .rename(columns={"<lambda_0>": "share_above_zero"})
    )
    vs.to_csv(os.path.join(args.out, "validity_split.csv"))

    # ---- 4. the seed spread, read against the gap it has to resolve ---- #
    present = [s for s in SEEDS if s in sims]
    lines = ["# Seed spread", ""]
    if len(present) >= 2:
        lines += [
            "The spread is the measurement, not a nuisance. Each row gives the "
            "three seeds, their range, and the gap the experiment is trying to "
            "resolve -- the baseline clone against the best rule. When the "
            "range exceeds that gap, a mean over three seeds is not a result.",
            "",
            "| statistic | "
            + " | ".join(present)
            + " | range | clone | best rule | gap |",
            "|---" * (len(present) + 5) + "|",
        ]
        best_rule = (
            out.drop(index=[r for r in present if r in out.index], errors="ignore")
            .drop(index=[BASELINE, "never"], errors="ignore")["common_good"]
            .idxmax()
            if len(out) > len(present) + 2
            else None
        )
        for col, src in (
            ("punish_rate", punish),
            ("mean_punishment", punish),
            ("common_good", out),
        ):
            vals = [float(src.loc[s, col]) for s in present if s in src.index]
            if len(vals) < 2:
                continue
            rng = max(vals) - min(vals)
            base = float(src.loc[BASELINE, col]) if BASELINE in src.index else np.nan
            br = (
                float(src.loc[best_rule, col])
                if best_rule and best_rule in src.index
                else np.nan
            )
            gap = abs(br - base) if not np.isnan(br) else np.nan
            lines.append(
                f"| {col} | "
                + " | ".join(f"{v:.4f}" for v in vals)
                + f" | **{rng:.4f}** | {base:.4f} | {br:.4f} | {gap:.4f} |"
            )
        lines += ["", f"Best rule by common good: `{best_rule}`."]
    with open(os.path.join(args.out, "seed_spread.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(punish.round(4).to_string(), "\n")
    print(out.round(3).to_string(), "\n")
    print(shape.round(3).to_string(), "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
