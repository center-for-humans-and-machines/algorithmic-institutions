"""When the population stopped disagreeing about the SHAPE of the policy.

A correction to `collapse.py`, and the correction matters. That script detects
saturation by counting members whose mean punishment is exactly 0. That
criterion finds seeds 42, 43 and 45 and misses seeds 44 and 46 entirely --
those two keep every member punishing and so look alive, while their evaluated
policy is a **constant**: punish everyone exactly 2 (s44) or exactly 1 (s46),
regardless of what they contributed. A flat tax is just as degenerate as a
zero policy for the question this arm exists to answer, and the
zero-punishment detector cannot see it.

The right detector is the spread of the per-member contribution-punishment
SLOPE. At generation 0 the 40 untrained members span −1.57 to +3.10 (slope sd
of order 1). If selection is choosing a direction for the contingency, that
spread should narrow around a non-zero mean. If selection is eliminating the
contingency, it narrows around zero. If nothing is happening, it stays.

Columns, per seed and evaluated generation:
  member_slope_sd    spread of the per-member slope -- the shape variance the
                     population still carries
  member_slope_mean  where that spread is centred
  theta_slope        the evaluated policy's own slope
  frac_human_sign    fraction of members with the human (negative) sign

Usage:
    python scripts/rl_es/shape_degeneracy.py --seeds 42,43,44,45,46
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
BIN_INDEX = {b: i for i, b in enumerate(RPA_LABELS)}


def slopes_by(frame, group_cols):
    """Least-squares slope of mean punishment across the six bins."""
    out = []
    for key, g in frame.groupby(group_cols):
        g = g.dropna(subset=["mean_punishment"])
        if len(g) < 2:
            continue
        x = g["contribution_bin"].map(BIN_INDEX).to_numpy(dtype=float)
        y = g["mean_punishment"].to_numpy(dtype=float)
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec["slope"] = float(np.polyfit(x, y, 1)[0])
        rec["level"] = float(y.mean())
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--subset", default="all", choices=("all", "valid"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        path = os.path.join(
            ROOT,
            f"artifacts/manager/rl_es_s{seed}/metrics/"
            f"rl_es_s{seed}_policy_shape.parquet",
        )
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        df = df[df["subset"] == args.subset]

        members = slopes_by(
            df[(df["sampling"] == "es-population") & (df["member"] >= 0)],
            ["update_step", "member"],
        )
        theta = slopes_by(df[df["sampling"] == "greedy"], ["update_step"]).set_index(
            "update_step"
        )
        for generation, g in members.groupby("update_step"):
            rows.append(
                {
                    "seed": seed,
                    "generation": int(generation),
                    "n_members": len(g),
                    "member_slope_sd": float(g["slope"].std()),
                    "member_slope_mean": float(g["slope"].mean()),
                    "member_slope_min": float(g["slope"].min()),
                    "member_slope_max": float(g["slope"].max()),
                    "member_level_sd": float(g["level"].std()),
                    "frac_human_sign": float((g["slope"] < 0).mean()),
                    "theta_slope": float(theta["slope"].get(generation, np.nan)),
                    "theta_level": float(theta["level"].get(generation, np.nan)),
                }
            )

    if not rows:
        print("no policy shape parquets found")
        return
    traj = pd.DataFrame(rows)
    traj.to_csv(os.path.join(args.out, "shape_degeneracy_trajectory.csv"), index=False)

    # When did the shape variance die? The first generation from which a
    # rolling mean of the member slope spread never again exceeds 10% of its
    # generation-0 value. Rolling rather than pointwise: the raw series blips
    # above the threshold for a single evaluated generation now and then, long
    # after the population has stopped carrying any real shape variance, and a
    # pointwise rule reports those blips as if the population had recovered.
    summary = []
    for seed, g in traj.groupby("seed"):
        g = g.sort_values("generation").reset_index(drop=True)
        start = float(g["member_slope_sd"].iloc[0])
        threshold = 0.1 * start
        smooth = g["member_slope_sd"].rolling(5, min_periods=1).mean().to_numpy()
        # FIRST CROSSING DOWN, not "never again exceeds". The raw series blips
        # back above the threshold for isolated evaluated generations long
        # after the population has stopped carrying real shape variance, and a
        # "never again" rule reports the last blip -- which moves by a thousand
        # generations if one member happens to differ once. The first crossing
        # is stable, and `frac_above_after` below says how much the series
        # recovers afterwards, which is the thing a reader actually wants.
        below = np.flatnonzero(smooth <= threshold)
        gen = int(g.loc[below[0], "generation"]) if len(below) else None
        after = smooth[below[0] :] if len(below) else np.array([])
        frac_above_after = float((after > threshold).mean()) if len(after) else np.nan
        summary.append(
            {
                "seed": seed,
                "slope_sd_generation_0": start,
                "slope_sd_final": float(g["member_slope_sd"].iloc[-1]),
                "shrinkage_factor": start
                / max(float(g["member_slope_sd"].iloc[-1]), 1e-9),
                "shape_degenerate_from_generation": gen,
                "frac_above_threshold_after": frac_above_after,
                "theta_slope_final": float(g["theta_slope"].iloc[-1]),
                "theta_level_final": float(g["theta_level"].iloc[-1]),
                "frac_human_sign_final": float(g["frac_human_sign"].iloc[-1]),
            }
        )
    out = pd.DataFrame(summary)
    out.to_csv(os.path.join(args.out, "shape_degeneracy.csv"), index=False)

    print("=== SHAPE DEGENERACY: when the population stopped disagreeing ===")
    print(out.round(4).to_string(index=False))
    print("\n=== member slope spread over training ===")
    piv = traj.pivot_table(index="generation", columns="seed", values="member_slope_sd")
    show = piv.loc[piv.index.isin([0, 100, 200, 400, 800, 1600, 3200, 3980])]
    print(show.round(4).to_string())


if __name__ == "__main__":
    main()
