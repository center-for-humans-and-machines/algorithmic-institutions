"""Assemble the finished arm's results into the tables the log reports.

Consumes what `guard.py shape --targeting-out` wrote per manager and emits:

  policy_shape.csv      mean punishment per RPA bin, every seed, arm and
                        control, with the human and clone columns and the row
                        count behind every mean
  shape_verdict.csv     one row per seed: the decisive numbers for "did the
                        human policy shape come back" -- the {0} minus {20}
                        contrast, whether the curve is monotone decreasing
                        across all six bins, and the rank correlation with
                        the human curve
  targeting.csv         leaver-minus-stayer contribution per seed
  paired.csv            arm minus control at the same seed, which is the
                        comparison the shared seeds were chosen for

    python scripts/rl_anneal_local/combine_results.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.evaluation_suite.metrics import RPA_LABELS  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/rl_anneal_local")
SEEDS = (42, 43, 44, 45, 46)
ARMS = {"arm": "rl_anneal_local_s{}", "control": "rl_new_clones_s{}"}

# The human curve, for the rank correlation. Read from the shape CSVs
# themselves rather than pasted, so it cannot drift from the source.


def load_shape(job):
    path = os.path.join(OUT, f"shape_{job}.csv")
    return pd.read_csv(path).set_index("contribution_bin").reindex(RPA_LABELS)


def verdict(curve, human):
    """The decisive numbers. The human managers punish free-riders hardest and
    full contributors least, so a recovered shape is decreasing: {0} minus
    {20} positive, monotone across all six bins, rank correlation +1 with the
    human curve. An inverted manager has {0} minus {20} negative."""
    diffs = curve.diff().dropna()
    return {
        "punish_at_0": curve[RPA_LABELS[0]],
        "punish_at_20": curve[RPA_LABELS[-1]],
        "zero_minus_twenty": curve[RPA_LABELS[0]] - curve[RPA_LABELS[-1]],
        "monotone_decreasing": bool((diffs <= 0).all()),
        "monotone_increasing": bool((diffs >= 0).all()),
        "spearman_vs_human": float(curve.corr(human, method="spearman")),
    }


def main():
    wide, verdicts, targeting = {}, [], []
    human = clone_ref = None

    for arm, tmpl in ARMS.items():
        for seed in SEEDS:
            job = tmpl.format(seed)
            sh = load_shape(job)
            if human is None:
                human = sh["human_mean"]
                wide["human"] = human
                wide["human_n"] = sh["human_n"]
            if clone_ref is None:
                clone_ref = sh["clone_mean"]
                wide["clone"] = clone_ref
                wide["clone_n"] = sh["clone_n"]
            key = f"{arm}_s{seed}"
            wide[key] = sh["evaluated_mean"]
            wide[f"{key}_n"] = sh["evaluated_n"]
            verdicts.append(
                {"arm": arm, "seed": seed, **verdict(sh["evaluated_mean"], human)}
            )

            t = pd.read_csv(os.path.join(OUT, f"targeting_{job}.csv"))
            own = t[t["manager"].str.startswith(job)].iloc[0]
            targeting.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "leaver_contribution": own["leaver_contribution"],
                    "stayer_contribution": own["stayer_contribution"],
                    "leaver_minus_stayer": own["leaver_minus_stayer"],
                    "leave_rate": own["leave_rate"],
                    "n_decisions": own["n_decisions"],
                }
            )

    shape = pd.DataFrame(wide).reindex(RPA_LABELS)
    shape.index.name = "contribution_bin"
    shape.to_csv(os.path.join(OUT, "policy_shape.csv"))

    v = pd.DataFrame(verdicts)
    v.to_csv(os.path.join(OUT, "shape_verdict.csv"), index=False)

    # the human and clone rows, on the same verdict columns, as the target
    ref = pd.DataFrame(
        [
            {"arm": "human", "seed": None, **verdict(human, human)},
            {"arm": "clone", "seed": None, **verdict(clone_ref, human)},
        ]
    )

    t = pd.DataFrame(targeting)
    t.to_csv(os.path.join(OUT, "targeting.csv"), index=False)

    # Paired: the shared seeds are the whole point, so arm minus control at
    # the same seed, never arm mean minus control mean.
    pv = v.pivot(index="seed", columns="arm")
    pt = t.pivot(index="seed", columns="arm")
    paired = pd.DataFrame(
        {
            "zero_minus_twenty_arm": pv[("zero_minus_twenty", "arm")],
            "zero_minus_twenty_control": pv[("zero_minus_twenty", "control")],
            "zero_minus_twenty_diff": pv[("zero_minus_twenty", "arm")]
            - pv[("zero_minus_twenty", "control")],
            "leaver_minus_stayer_arm": pt[("leaver_minus_stayer", "arm")],
            "leaver_minus_stayer_control": pt[("leaver_minus_stayer", "control")],
            "leaver_minus_stayer_diff": pt[("leaver_minus_stayer", "arm")]
            - pt[("leaver_minus_stayer", "control")],
        }
    )
    paired.to_csv(os.path.join(OUT, "paired.csv"))

    pd.set_option("display.width", 200)
    print("=== policy shape (mean punishment per contribution bin) ===")
    print(
        shape[[c for c in shape.columns if not c.endswith("_n")]].round(2).to_string()
    )
    print()
    print("=== row counts ===")
    print(shape[[c for c in shape.columns if c.endswith("_n")]].to_string())
    print()
    print("=== shape verdict, per seed (reference rows first) ===")
    print(pd.concat([ref, v], ignore_index=True).round(3).to_string(index=False))
    print()
    print("=== targeting: leavers minus stayers, in contribution ===")
    print(t.round(3).to_string(index=False))
    print()
    print("=== paired, arm minus control at the same seed ===")
    print(paired.round(3).to_string())


if __name__ == "__main__":
    main()
