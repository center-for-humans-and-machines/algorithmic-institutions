"""Pairwise win-rate tables from a sim's per_round.parquet.

For each matchup (e.g. rule_k1 vs zero, rule_k1 vs ah) reports the fraction
of episodes side a beats side b on a per-group aggregate of a metric,
averaged across the rounds of the episode. Both position assignments
(`a_vs_b` and `b_vs_a`) are pooled, so the result is symmetric in group
label. A same-manager pairing (`x_vs_x`) reports group 0 vs group 1.

Per-group aggregation is selectable with --agg:
  - sum : per round, SUM the metric over the agents in the group
  - mean: per round, MEAN the metric over the agents in the group

EMPTY GROUP-ROUNDS COUNT AS 0. A group with no members that round
produced nothing, so its per-round value is 0 and it stays in the
denominator when averaging across the 24 rounds. This is applied
identically for sum and mean, and is the whole point of this script:
a manager that empties its group is penalised, not excused.

Three metrics are tabled by default (payoff, common_good, contribution).

Sides are read from the pairing name `<g0>_vs_<g1>`; a matchup is oriented
as its first pairing in the file.

Usage:
    python scripts/plotting/plot_winrates.py <sim_dir> \\
        [--agg sum|mean] \\
        [--metrics payoff common_good contribution] \\
        [--out tables.md]

Example:
    python scripts/plotting/plot_winrates.py \\
        plots/simulation/19_2g8a_rule_based_vs_zero --agg mean
"""

import argparse
import os
import sys

import pandas as pd

DEFAULT_METRICS = ["payoff", "common_good", "contribution"]


def load_per_round(sim_dir: str) -> pd.DataFrame:
    path = os.path.join(sim_dir, "per_round.parquet")
    if not os.path.exists(path):
        sys.exit(f"per_round.parquet not found at {path}")
    return pd.read_parquet(path)


def pairing_sides(df: pd.DataFrame) -> dict:
    """Map run -> (matchup, group of side a).

    Sides come from the pairing name `<g0>_vs_<g1>`. A matchup is oriented
    as its first pairing in the file (config order), so `x_vs_y` and
    `y_vs_x` pool into one matchup; for `x_vs_x`, side a is group 0.
    """
    sides, orient = {}, {}
    for run in df["run"].unique():
        pairing = run.split("managed by ")[-1]
        g0, g1 = pairing.split("_vs_")
        a, b = orient.setdefault(frozenset((g0, g1)), (g0, g1))
        sides[run] = (f"{a} vs {b}", 0 if g0 == a else 1)
    return sides


def episode_scores(sub: pd.DataFrame, a_g: int, metric: str, agg: str):
    """Return (a_score, b_score) Series indexed by episode.

    Per (episode, round, group) aggregate the metric over agents (sum or
    mean), zero-fill empty group-rounds, then average across rounds.
    """
    grouped = sub.groupby(["episode", "round_number", "agent_group"])[metric]
    per_round = grouped.sum() if agg == "sum" else grouped.mean()
    piv = per_round.reset_index().pivot_table(
        index=["episode", "round_number"], columns="agent_group", values=metric
    )
    piv = piv.reindex(columns=[0, 1]).fillna(0.0)  # empty group-round -> 0
    ep = piv.groupby("episode").mean()  # average across the episode's rounds
    return ep[a_g], ep[1 - a_g]


def empty_fraction(sub: pd.DataFrame, group: int) -> float:
    """Mean fraction of rounds (over episodes) the group had no members."""
    counts = sub.groupby(["episode", "round_number", "agent_group"]).size()
    cpiv = counts.reset_index(name="n").pivot_table(
        index=["episode", "round_number"], columns="agent_group", values="n"
    )
    cpiv = cpiv.reindex(columns=[0, 1])
    return float(cpiv[group].isna().mean())


def winrate_table(df: pd.DataFrame, sides: dict, metric: str, agg: str):
    """Per-matchup win-rate rows pooled over both position assignments."""
    rows = []
    for run, (matchup, a_g) in sides.items():
        sub = df[df["run"] == run]
        a, b = episode_scores(sub, a_g, metric, agg)
        a_empty = empty_fraction(sub, a_g)
        b_empty = empty_fraction(sub, 1 - a_g)
        for epid in a.index:
            rows.append(
                {
                    "matchup": matchup,
                    "a": a[epid],
                    "b": b[epid],
                    "a_empty": a_empty,
                    "b_empty": b_empty,
                }
            )
    res = pd.DataFrame(rows)
    out = []
    for key in res["matchup"].unique():
        s = res[res["matchup"] == key]
        out.append(
            {
                "matchup (a vs b)": key,
                "episodes": len(s),
                "a_win%": round(100 * float((s["a"] > s["b"]).mean()), 1),
                "b_win%": round(100 * float((s["b"] > s["a"]).mean()), 1),
                "a_mean": round(float(s["a"].mean()), 2),
                "b_mean": round(float(s["b"].mean()), 2),
                "a_empty%": round(100 * float(s["a_empty"].mean()), 1),
                "b_empty%": round(100 * float(s["b_empty"].mean()), 1),
            }
        )
    return pd.DataFrame(out)


def to_markdown(tbl: pd.DataFrame) -> str:
    cols = list(tbl.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in tbl.itertuples(index=False)
    ]
    return "\n".join([head, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", help="Sim output dir with per_round.parquet")
    parser.add_argument(
        "--agg",
        choices=["sum", "mean"],
        default="sum",
        help="Per-group aggregation over agents each round (default sum)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help=f"Metrics to table (default: {' '.join(DEFAULT_METRICS)})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional markdown file to write the tables to",
    )
    args = parser.parse_args()

    df = load_per_round(args.sim_dir)
    sides = pairing_sides(df)

    blocks = [
        f"# Win rates ({args.agg} per group, empty round = 0)",
        f"_Source: {os.path.join(args.sim_dir, 'per_round.parquet')} — "
        f"{len(sides)} runs pooled by matchup._",
    ]
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"skipping {metric!r}: not a column", file=sys.stderr)
            continue
        tbl = winrate_table(df, sides, metric, args.agg)
        blocks.append(f"\n## {metric}\n\n{to_markdown(tbl)}")

    text = "\n".join(blocks)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
