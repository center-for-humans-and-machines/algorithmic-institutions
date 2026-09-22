"""Print the per-pairing sum-payoff winner table as a markdown table.

For each pairing, computes the per-round mean of payoff_sum across episodes
(the line shown in the sim trajectory plots), then reports the avg and max
of that 24-point trend per group along with the winner by avg.

Usage:
    python scripts/plotting/plot_sim_winner_finder.py <sim_dir>

Example:
    python scripts/plotting/plot_sim_winner_finder.py plots/simulation/19_2g8a_rule_based_vs_zero
"""
import argparse
import os
import sys

import pandas as pd


def parse_sides(pairing: str) -> tuple:
    parts = pairing.split("_vs_")
    if len(parts) == 2:
        return parts[0], parts[1]
    return "g0", "g1"


def build_table(sim_dir: str) -> pd.DataFrame:
    path = os.path.join(sim_dir, "per_round.parquet")
    if not os.path.exists(path):
        sys.exit(f"per_round.parquet not found at {path}")
    df = pd.read_parquet(path)
    df["pairing"] = df["run"].str.replace(
        "ah group_switching managed by ", "", regex=False
    )
    df["pairing"] = df["pairing"].str.replace("ah managed by ", "", regex=False)

    key = ["pairing", "episode", "round_number", "group_id"]
    ps = (
        df.groupby(key)["payoff"]
        .sum()
        .reset_index()
        .rename(columns={"payoff": "payoff_sum"})
    )
    round_mean = (
        ps.groupby(["pairing", "group_id", "round_number"])["payoff_sum"]
        .mean()
        .reset_index()
    )
    agg = (
        round_mean.groupby(["pairing", "group_id"])["payoff_sum"]
        .agg(["mean", "max"])
        .reset_index()
    )

    rows = []
    for p in df["pairing"].unique():
        m0, m1 = parse_sides(p)
        g0 = agg[(agg.pairing == p) & (agg.group_id == 0)].iloc[0]
        g1 = agg[(agg.pairing == p) & (agg.group_id == 1)].iloc[0]
        if g0["mean"] > g1["mean"]:
            winner = f"g0 ({m0})"
        elif g1["mean"] > g0["mean"]:
            winner = f"g1 ({m1})"
        else:
            winner = "tie"
        rows.append(
            {
                "pairing": p,
                "g0 mgr": m0,
                "g0 avg": round(g0["mean"], 1),
                "g0 max": round(g0["max"], 1),
                "g1 mgr": m1,
                "g1 avg": round(g1["mean"], 1),
                "g1 max": round(g1["max"], 1),
                "winner": winner,
            }
        )
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    aligns = ["---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(aligns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([header, sep] + rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", help="Sim output directory with per_round.parquet")
    args = parser.parse_args()
    tbl = build_table(args.sim_dir)
    print(df_to_markdown(tbl))


if __name__ == "__main__":
    main()
