"""The per-bin behaviour-versus-evaluation shift, and the decorrelation
prediction it tests.

The aggregate gap ("the behaviour policy punishes 1.7 to 6.6 times as hard as
the policy being evaluated") is a summary. The sharper statement, and the one
with no free parameters, is per contribution bin:

    epsilon-greedy over 31 uniform levels drags each bin toward the uniform
    mean of 15 by exactly  eps * (15 - evaluated_bin).

On the control arm that prediction holds -- regression slope 0.891 against the
predicted shift, sign flipping at precisely the one bin where the evaluated
policy already punishes above 15, and the buffer's shape flattened 9.4%
relative to what is evaluated. The annealed arm cuts the mean absolute per-bin
shift from 0.886 to 0.067.

THIS ARM'S PREDICTION IS DIFFERENT, AND SAYING SO PRECISELY IS THE POINT.
There is no action noise here, so there is nothing to drag anything toward 15.
But the shift is not zero either: the behaviour policies are theta perturbed
by sigma, so they differ from theta in whatever direction the perturbation
happens to point. The discriminating statistic is therefore not the size of
the shift but its RELATION TO 15:

    * `slope_vs_uniform_pull` -- regression of the observed per-bin shift on
      (15 - evaluated_bin). Near `eps` for an epsilon-greedy arm, near ZERO
      here, and that difference is the mechanism, not the magnitude.
    * `mean_abs_shift` -- comparable with the sibling arms' 0.886 / 0.067.
    * `flattening` -- the behaviour shape's slope across bins relative to the
      evaluated shape's. Uniform action noise flattens (drags every bin to the
      same 15); parameter noise has no reason to.

Usage:
    python scripts/rl_es/behaviour_shift.py --seeds 42,43,44,45,46
    python scripts/rl_es/behaviour_shift.py --shape <policy_shape.parquet>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.evaluation_suite.metrics import RPA_LABELS  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/evaluation/rl_manager_es")

#: The mean of a uniform draw over the 31 ordinal punishment levels 0..30.
#: What epsilon-greedy pulls every bin towards, and the anchor of the
#: prediction this script tests.
UNIFORM_MEAN = 15.0


def shift_rows(shape, subset="all"):
    """One row per (generation, bin): what was run against what was scored."""
    df = shape[shape["subset"] == subset]
    behaviour = df[(df["sampling"] == "es-population") & (df["member"] == -1)]
    evaluated = df[df["sampling"] == "greedy"]
    key = ["update_step", "contribution_bin"]
    merged = behaviour.set_index(key)[["mean_punishment", "n"]].join(
        evaluated.set_index(key)[["mean_punishment"]],
        lsuffix="_behaviour",
        rsuffix="_evaluated",
        how="inner",
    )
    merged = merged.reset_index().rename(columns={"n": "n_behaviour"})
    merged["shift"] = (
        merged["mean_punishment_behaviour"] - merged["mean_punishment_evaluated"]
    )
    merged["predicted_uniform_pull"] = (
        UNIFORM_MEAN - merged["mean_punishment_evaluated"]
    )
    return merged


def slope_across_bins(frame, column):
    """Least-squares slope of `column` across the six contribution bins."""
    order = {b: i for i, b in enumerate(RPA_LABELS)}
    f = frame.dropna(subset=[column])
    if len(f) < 2:
        return np.nan
    x = f["contribution_bin"].map(order).to_numpy(dtype=float)
    return float(np.polyfit(x, f[column].to_numpy(dtype=float), 1)[0])


def summarise(rows, label):
    out = []
    for generation, frame in rows.groupby("update_step"):
        pred = frame["predicted_uniform_pull"].to_numpy(dtype=float)
        obs = frame["shift"].to_numpy(dtype=float)
        ok = ~(np.isnan(pred) | np.isnan(obs))
        # Regression through the origin: the prediction has no intercept
        # (a bin already at 15 is not pulled anywhere).
        denom = float((pred[ok] ** 2).sum())
        slope = float((pred[ok] * obs[ok]).sum() / denom) if denom else np.nan
        # The raw slope conflates two things: a uniform shift in the LEVEL of
        # punishment, and a shift in the SHAPE across bins. When the evaluated
        # policy is flat, `pred` is near-constant and the raw slope degenerates
        # to (mean shift) / (mean pull) -- it measures the level and tests
        # nothing. Demeaning both sides isolates the bin-to-bin pattern, which
        # is what the decorrelation claim is actually about: uniform action
        # noise pulls the bins FURTHEST from 15 hardest, and so flattens the
        # shape. Read the demeaned slope when `evaluated_bin_spread` is small.
        pc, oc = pred[ok] - pred[ok].mean(), obs[ok] - obs[ok].mean()
        dd = float((pc**2).sum())
        slope_demeaned = float((pc * oc).sum() / dd) if dd else np.nan
        eval_slope = slope_across_bins(frame, "mean_punishment_evaluated")
        beh_slope = slope_across_bins(frame, "mean_punishment_behaviour")
        out.append(
            {
                "run": label,
                "generation": int(generation),
                "mean_abs_shift": float(np.nanmean(np.abs(obs))),
                "max_abs_shift": float(np.nanmax(np.abs(obs))),
                "slope_vs_uniform_pull": slope,
                "slope_vs_uniform_pull_demeaned": slope_demeaned,
                # How much the evaluated policy varies across bins. The slope
                # statistics are uninformative when this is ~0.
                "evaluated_bin_spread": float(np.nanstd(pred[ok])),
                "evaluated_shape_slope": eval_slope,
                "behaviour_shape_slope": beh_slope,
                "flattening": (
                    float(1.0 - beh_slope / eval_slope)
                    if eval_slope not in (0.0, np.nan) and not np.isnan(eval_slope)
                    else np.nan
                ),
                "evaluated_mean_punishment": float(
                    np.nanmean(frame["mean_punishment_evaluated"])
                ),
                "behaviour_mean_punishment": float(
                    np.nanmean(frame["mean_punishment_behaviour"])
                ),
            }
        )
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--shape", default=None, help="a single policy_shape parquet")
    ap.add_argument("--subset", default="all", choices=("all", "valid"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sources = {}
    if args.shape:
        sources[os.path.basename(args.shape)] = args.shape
    else:
        for seed in [int(s) for s in args.seeds.split(",")]:
            path = os.path.join(
                ROOT,
                f"artifacts/manager/rl_es_s{seed}/metrics/"
                f"rl_es_s{seed}_policy_shape.parquet",
            )
            if os.path.exists(path):
                sources[f"rl_es_s{seed}"] = path

    per_bin, summaries = [], []
    for label, path in sources.items():
        rows = shift_rows(pd.read_parquet(path), args.subset)
        rows["run"] = label
        per_bin.append(rows)
        summaries.append(summarise(rows, label))
    if not per_bin:
        print("no policy shape parquets found yet")
        return

    pd.concat(per_bin).to_csv(
        os.path.join(args.out, "behaviour_shift_per_bin.csv"), index=False
    )
    summary = pd.concat(summaries)
    summary.to_csv(os.path.join(args.out, "behaviour_shift.csv"), index=False)
    cols = [
        "run",
        "generation",
        "mean_abs_shift",
        "slope_vs_uniform_pull",
        "slope_vs_uniform_pull_demeaned",
        "evaluated_bin_spread",
        "evaluated_shape_slope",
        "behaviour_shape_slope",
        "evaluated_mean_punishment",
        "behaviour_mean_punishment",
    ]
    print(summary[cols].round(4).to_string(index=False))
    print(
        "\nslope_vs_uniform_pull is the discriminator: ~eps for an "
        "epsilon-greedy arm (0.1 for the control), ~0 here if the shift is "
        "parameter noise with no pull toward the uniform mean of 15.\n"
        "Read the demeaned column when evaluated_bin_spread is small: with a "
        "flat evaluated policy the raw slope is just (level shift)/(15 - level)"
        " and tests nothing."
    )


if __name__ == "__main__":
    main()
