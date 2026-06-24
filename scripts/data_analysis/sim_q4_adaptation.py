"""Q4 (post-switch adaptation) event study on simulation per_round data.

Replicates Q4 of reports/human_behavior_analysis_50ep.md on a sim's
per_round.parquet: for every switch (agent_group changes between rounds),
regress the switcher's post-switch contribution on their own pre-switch
contribution and the new group's leave-one-out peer mean at arrival
(both standardised). A positive new-group peer coefficient means the AHs
adopt the new group's norm (conditional cooperation), as humans do.

The human 50ep reference is printed alongside for comparison:
  own pre-switch +0.45, new-group peers +0.18 (raw 7.70 -> 9.70 -> 9.40).

Pools all switches across runs/episodes in each sim dir.

Usage:
    python scripts/data_analysis/sim_q4_adaptation.py <sim_dir> [<sim_dir> ...]

Example:
    python scripts/data_analysis/sim_q4_adaptation.py \\
        plots/simulation/19_2g8a_rule_based_vs_zero \\
        plots/simulation/19_2g8a_rule_based_vs_zero_same_group
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HUMAN = {
    "n": 547,
    "own_pre": 0.45,
    "new_peers": 0.18,
    "raw_pre": 7.70,
    "raw_peers": 9.70,
    "raw_post": 9.40,
}


def stdz(x):
    return (x - x.mean()) / x.std()


def ols(df, feats, tgt):
    """Standardised OLS; returns (coef dict, n)."""
    a = df.dropna(subset=feats + [tgt])
    x = stdz(a[feats]).copy()
    x.insert(0, "const", 1.0)
    y = stdz(a[tgt]).values
    b, *_ = np.linalg.lstsq(x.values, y, rcond=None)
    return dict(zip(feats, b[1:])), len(a)


def q4(per_round_path):
    df = pd.read_parquet(per_round_path)
    df = df.sort_values(["run", "episode", "participant_code", "round_number"])
    g = df.groupby(["run", "episode", "participant_code"])
    df["pre"] = g["contribution"].shift(1)
    df["prev_group"] = g["agent_group"].shift(1)
    df["switch"] = df["prev_group"].notna() & df["prev_group"].ne(df["agent_group"])
    # leave-one-out mean contribution of the (new) group at the arrival round
    gr = df.groupby(["run", "episode", "round_number", "agent_group"])["contribution"]
    gsum, gcnt = gr.transform("sum"), gr.transform("count")
    df["new_peers"] = (gsum - df["contribution"]) / (gcnt - 1)
    sw = df[df["switch"]].rename(columns={"contribution": "post"})
    sw = sw.dropna(subset=["pre", "post", "new_peers"])
    betas, n = ols(sw, ["pre", "new_peers"], "post")
    return {
        "n": n,
        "own_pre": betas["pre"],
        "new_peers": betas["new_peers"],
        "raw_pre": sw["pre"].mean(),
        "raw_peers": sw["new_peers"].mean(),
        "raw_post": sw["post"].mean(),
    }


def fmt(label, r):
    return (
        f"{label:<26} {r['n']:>6}  {r['own_pre']:+.3f}     {r['new_peers']:+.3f}"
        f"      {r['raw_pre']:.2f} -> {r['raw_peers']:.2f} -> {r['raw_post']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sim_dirs",
        nargs="+",
        help="Sim output dir(s) containing per_round.parquet",
    )
    args = parser.parse_args()

    print(f"{'source':<26} {'N':>6}  own_pre   new_peers   raw: pre -> peers -> post")
    print(fmt("HUMAN (50ep)", HUMAN))
    for d in args.sim_dirs:
        path = os.path.join(d, "per_round.parquet")
        if not os.path.exists(path):
            print(f"[skip] no per_round.parquet in {d}", file=sys.stderr)
            continue
        print(fmt(os.path.basename(os.path.normpath(d)), q4(path)))


if __name__ == "__main__":
    main()
