"""Rule-vs-zero win-rate tables from a sim's per_round.parquet.

For each rule strength (e.g. k1/k4/k8) reports the fraction of episodes
the rule-managed group beats the zero-punishment group on a per-group
aggregate of a metric, averaged across the rounds of the episode. Both
position assignments (rule as group 0 and rule as group 1) are pooled,
so the result is symmetric in group label.

Per-group aggregation is selectable with --agg:
  - sum : per round, SUM the metric over the agents in the group
  - mean: per round, MEAN the metric over the agents in the group

EMPTY GROUP-ROUNDS COUNT AS 0. A group with no members that round
produced nothing, so its per-round value is 0 and it stays in the
denominator when averaging across the 24 rounds. This is applied
identically for sum and mean, and is the whole point of this script:
a manager that empties its group is penalised, not excused.

Three metrics are tabled by default (payoff, common_good, contribution).

The rule group of each run is detected as the agent_group with the
larger mean punishment (the zero manager never punishes); runs with no
punisher at all (e.g. zero_vs_zero) are skipped. The matchup key is the
``k<N>`` token parsed from the run/pairing name.

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
import re
import sys

import pandas as pd

DEFAULT_METRICS = ["payoff", "common_good", "contribution"]


def load_per_round(sim_dir: str) -> pd.DataFrame:
    path = os.path.join(sim_dir, "per_round.parquet")
    if not os.path.exists(path):
        sys.exit(f"per_round.parquet not found at {path}")
    return pd.read_parquet(path)


def detect_rule_group(df: pd.DataFrame) -> dict:
    """Map run -> agent_group of the rule (punishing) manager.

    The zero manager never punishes, so the group with the larger mean
    punishment is the rule group. Runs with no punisher are skipped.
    """
    pun = (
        df.groupby(["run", "agent_group"])["punishment"].mean().unstack(fill_value=0.0)
    )
    rule_of = {}
    for run, row in pun.iterrows():
        if row.max() <= 1e-9:
            continue  # no punisher in this run (e.g. zero_vs_zero)
        rule_of[run] = int(row.idxmax())
    return rule_of


def matchup_key(run: str) -> str:
    """Position-independent matchup label, e.g. 'k1' from the run name."""
    pairing = run.split("managed by ")[-1]
    m = re.search(r"k(\d+)", pairing)
    return f"k{m.group(1)}" if m else pairing


def episode_scores(sub: pd.DataFrame, rule_g: int, metric: str, agg: str):
    """Return (rule_score, zero_score) Series indexed by episode.

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
    return ep[rule_g], ep[1 - rule_g]


def empty_fraction(sub: pd.DataFrame, group: int) -> float:
    """Mean fraction of rounds (over episodes) the group had no members."""
    counts = sub.groupby(["episode", "round_number", "agent_group"]).size()
    cpiv = counts.reset_index(name="n").pivot_table(
        index=["episode", "round_number"], columns="agent_group", values="n"
    )
    cpiv = cpiv.reindex(columns=[0, 1])
    return float(cpiv[group].isna().mean())


def winrate_table(df: pd.DataFrame, rule_of: dict, metric: str, agg: str):
    """Per-matchup win-rate rows pooled over both position assignments."""
    rows = []
    for run, rule_g in rule_of.items():
        sub = df[df["run"] == run]
        rule, zero = episode_scores(sub, rule_g, metric, agg)
        re_empty = empty_fraction(sub, rule_g)
        ze_empty = empty_fraction(sub, 1 - rule_g)
        for epid in rule.index:
            rows.append(
                {
                    "matchup": matchup_key(run),
                    "rule": rule[epid],
                    "zero": zero[epid],
                    "rule_empty": re_empty,
                    "zero_empty": ze_empty,
                }
            )
    res = pd.DataFrame(rows)
    out = []
    for key in sorted(res["matchup"].unique()):
        s = res[res["matchup"] == key]
        n = len(s)
        out.append(
            {
                "matchup": f"{key} vs zero",
                "episodes": n,
                "rule_win%": round(100 * float((s["rule"] > s["zero"]).mean()), 1),
                "zero_win%": round(100 * float((s["zero"] > s["rule"]).mean()), 1),
                "rule_mean": round(float(s["rule"].mean()), 2),
                "zero_mean": round(float(s["zero"].mean()), 2),
                "rule_empty%": round(100 * float(s["rule_empty"].mean()), 1),
                "zero_empty%": round(100 * float(s["zero_empty"].mean()), 1),
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
    rule_of = detect_rule_group(df)
    if not rule_of:
        sys.exit("No punishing (rule) manager found in any run.")

    blocks = [
        f"# Win rates — rule vs zero ({args.agg} per group, empty round = 0)",
        f"_Source: {os.path.join(args.sim_dir, 'per_round.parquet')} — "
        f"{len(rule_of)} runs pooled by matchup._",
    ]
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"skipping {metric!r}: not a column", file=sys.stderr)
            continue
        tbl = winrate_table(df, rule_of, metric, args.agg)
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
