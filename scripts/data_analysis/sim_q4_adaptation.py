"""Q4 (post-switch adaptation) event study on simulation per_round data.

Q4 event study on a sim's per_round.parquet: for every switch (agent_group
changes between rounds), regress the switcher's post-switch contribution on
their own pre-switch contribution and the new group's mean contribution at the
PREVIOUS round (t-1) -- the level the switcher could actually observe when
choosing at arrival -- both standardised. A positive new-group peer coefficient
means the AHs adopt the new group's norm (conditional cooperation), as humans do.

The new-group peer term is the strictly-causal lagged level (t-1), not the
concurrent arrival-round mean; the human 50ep reference below is recomputed on
the same lagged definition (the report's concurrent Q4 was +0.18).

The human 50ep reference is printed alongside for comparison:
  own pre-switch +0.45, new-group peers +0.25 (raw 7.73 -> 10.50 -> 9.46).

Pools all switches across runs/episodes in each sim dir, unless --per-matchup
is given, in which case Q4 is reported separately for each run (matchup) -- e.g.
rule_k1_vs_zero / rule_k4_vs_zero / rule_k8_vs_zero against zero.

Usage:
    python scripts/data_analysis/sim_q4_adaptation.py <sim_dir> [<sim_dir> ...]
    python scripts/data_analysis/sim_q4_adaptation.py --per-matchup <sim_dir>

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

# Human 50ep reference, recomputed on the lagged (t-1) new-group peer mean
# (own pre-switch and post unchanged from the concurrent Q4; only the peer term
# and its raw level move: +0.18 -> +0.25, raw peers 9.70 -> 10.50).
HUMAN = {
    "n": 539,
    "own_pre": 0.446,
    "new_peers": 0.247,
    "raw_pre": 7.73,
    "raw_peers": 10.50,
    "raw_post": 9.46,
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


def build_switches(per_round_path):
    """Per-switch frame: one row per arrival round, with pre/post/new_peers."""
    df = pd.read_parquet(per_round_path)
    df = df.sort_values(["run", "episode", "participant_code", "round_number"])
    g = df.groupby(["run", "episode", "participant_code"])
    df["pre"] = g["contribution"].shift(1)
    df["prev_group"] = g["agent_group"].shift(1)
    df["switch"] = df["prev_group"].notna() & df["prev_group"].ne(df["agent_group"])
    # new group's mean contribution at the PREVIOUS round (t-1) -- the level the
    # switcher could observe when choosing at arrival. The switcher was in the
    # old group at t-1, so this is the new group's full mean (no leave-one-out).
    gmean = (
        df.groupby(["run", "episode", "round_number", "agent_group"])["contribution"]
        .mean()
        .to_dict()
    )
    df["new_peers"] = [
        gmean.get((run, ep, rn - 1, ag), np.nan)
        for run, ep, rn, ag in df[
            ["run", "episode", "round_number", "agent_group"]
        ].values
    ]
    sw = df[df["switch"]].rename(columns={"contribution": "post"})
    return sw.dropna(subset=["pre", "post", "new_peers"])


def q4_betas(sw):
    """Standardised Q4 regression on a switch frame -> result dict."""
    betas, n = ols(sw, ["pre", "new_peers"], "post")
    return {
        "n": n,
        "own_pre": betas["pre"],
        "new_peers": betas["new_peers"],
        "raw_pre": sw["pre"].mean(),
        "raw_peers": sw["new_peers"].mean(),
        "raw_post": sw["post"].mean(),
    }


def q4(per_round_path):
    return q4_betas(build_switches(per_round_path))


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
    parser.add_argument(
        "--per-matchup",
        action="store_true",
        help="Break Q4 down per run (matchup) instead of pooling all switches",
    )
    args = parser.parse_args()

    print(f"{'source':<26} {'N':>6}  own_pre   new_peers   raw: pre -> peers -> post")
    print(fmt("HUMAN (50ep)", HUMAN))
    for d in args.sim_dirs:
        path = os.path.join(d, "per_round.parquet")
        if not os.path.exists(path):
            print(f"[skip] no per_round.parquet in {d}", file=sys.stderr)
            continue
        name = os.path.basename(os.path.normpath(d))
        if args.per_matchup:
            sw = build_switches(path)
            print(f"\n{name}:")
            for run, sub in sw.groupby("run"):
                label = "  " + str(run).replace("ah group_switching managed by ", "")
                print(fmt(label, q4_betas(sub)))
        else:
            print(fmt(name, q4(path)))


if __name__ == "__main__":
    main()
