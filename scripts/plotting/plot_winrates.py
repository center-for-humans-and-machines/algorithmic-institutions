"""Win-rate tables from one or more sims' per_round.parquet.

By default writes the five win definitions of #226:

  1. Payoff sum, head to head: per round, the group's payoff summed over its
     members; the higher episode average wins the episode.
  2. Per-capita contribution, head to head: per round, the mean contribution
     per member.
  3. Per-capita common good, head to head: per round, the mean common good
     per member.
  4. Levin's criterion (#219), against an anchor: per round, the group's
     common pool 1.6*sum(c) - sum(p). Managers never meet; each plays the
     anchor (default `ah`) and the one whose group does best against it wins.
     Margins are taken against a reference manager's group against the anchor
     (default `zero`), with a 95% bootstrap interval over episodes. Only
     managers that played the anchor can be compared: any that did not are
     listed in a warning.
  5. Common pool, head to head: the same pool, the two groups of the same
     episode; win % and the paired mean margin with a 95% bootstrap interval.

Head-to-head definitions (1, 2, 3, 5) pool both position assignments
(`a_vs_b` and `b_vs_a`) into one matchup; a same-manager pairing (`x_vs_x`)
reports group 0 vs group 1. Sides are read from the pairing name
`<g0>_vs_<g1>`; a matchup is oriented as its first pairing in the file.

EMPTY GROUP-ROUNDS COUNT AS 0. A group with no members that round produced
nothing, so its per-round value is 0 and it stays in the denominator when
averaging across the episode's rounds: a manager that empties its group is
penalised, not excused.

Several sim dirs can be passed; their runs are pooled by matchup. This lets
definition 4 take its reference from another run (e.g. zero vs AH in run 1).

Custom tables: passing --agg or --metrics writes the per-metric head-to-head
tables instead (the format before the five definitions).

Usage:
    python scripts/plotting/plot_winrates.py <sim_dir> [<sim_dir> ...] \\
        [--anchor ah] [--reference zero] [--out tables.md]
    python scripts/plotting/plot_winrates.py <sim_dir> \\
        --agg sum|mean [--metrics payoff common_good contribution]

Example:
    python scripts/plotting/plot_winrates.py \\
        plots/simulation/25_LEVIN_run1_ah_zero_pairings \\
        plots/simulation/25_LEVIN_run3_sigmoid_pairings \\
        --out plots/simulation/25_LEVIN_run3_sigmoid_pairings/winrates.md
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_METRICS = ["payoff", "common_good", "contribution"]
POOL = "pool"  # per-agent 1.6*c - p; its group sum is the common pool
N_BOOT = 10000
BOOT_SEED = 0

# (title, metric, per-group aggregation, what the score is)
HEAD_TO_HEAD = [
    (
        "1. Payoff sum, head to head",
        "payoff",
        "sum",
        "Per round, the group's payoff summed over its members",
    ),
    (
        "2. Per-capita contribution, head to head",
        "contribution",
        "mean",
        "Per round, the mean contribution per member",
    ),
    (
        "3. Per-capita common good, head to head",
        "common_good",
        "mean",
        "Per round, the mean common good per member",
    ),
]


def load_per_round(sim_dirs: list) -> pd.DataFrame:
    frames = []
    for sim_dir in sim_dirs:
        path = os.path.join(sim_dir, "per_round.parquet")
        if not os.path.exists(path):
            sys.exit(f"per_round.parquet not found at {path}")
        df = pd.read_parquet(path)
        if len(sim_dirs) > 1:
            # keep runs of different sims apart; the pairing name stays last
            df["run"] = sim_dir + " | " + df["run"]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df[POOL] = 1.6 * df["contribution"] - df["punishment"]
    return df


def pairing_of(run: str) -> tuple:
    return tuple(run.split("managed by ")[-1].split("_vs_"))


def pairing_sides(df: pd.DataFrame) -> dict:
    """Map run -> (matchup, group of side a).

    Sides come from the pairing name `<g0>_vs_<g1>`. A matchup is oriented
    as its first pairing in the file (config order), so `x_vs_y` and
    `y_vs_x` pool into one matchup; for `x_vs_x`, side a is group 0.
    """
    sides, orient = {}, {}
    for run in df["run"].unique():
        g0, g1 = pairing_of(run)
        a, b = orient.setdefault(frozenset((g0, g1)), (g0, g1))
        sides[run] = (f"{a} vs {b}", 0 if g0 == a else 1)
    return sides


def group_round_scores(sub: pd.DataFrame, metric: str, agg: str) -> pd.DataFrame:
    """Per (episode, round): each group's aggregate, empty group-round = 0."""
    grouped = sub.groupby(["episode", "round_number", "agent_group"])[metric]
    per_round = grouped.sum() if agg == "sum" else grouped.mean()
    piv = per_round.reset_index().pivot_table(
        index=["episode", "round_number"], columns="agent_group", values=metric
    )
    return piv.reindex(columns=[0, 1]).fillna(0.0)  # empty group-round -> 0


def episode_scores(sub: pd.DataFrame, a_g: int, metric: str, agg: str):
    """Return (a_score, b_score) Series indexed by episode.

    Per (episode, round, group) aggregate the metric over agents (sum or
    mean), zero-fill empty group-rounds, then average across rounds.
    """
    ep = group_round_scores(sub, metric, agg).groupby("episode").mean()
    return ep[a_g], ep[1 - a_g]


def episode_members(sub: pd.DataFrame, group: int) -> pd.Series:
    """Mean group size per episode, empty group-round = 0."""
    counts = sub.groupby(["episode", "round_number", "agent_group"]).size()
    cpiv = counts.reset_index(name="n").pivot_table(
        index=["episode", "round_number"], columns="agent_group", values="n"
    )
    cpiv = cpiv.reindex(columns=[0, 1]).fillna(0.0)
    return cpiv[group].groupby("episode").mean()


def empty_fraction(sub: pd.DataFrame, group: int) -> float:
    """Mean fraction of rounds (over episodes) the group had no members."""
    counts = sub.groupby(["episode", "round_number", "agent_group"]).size()
    cpiv = counts.reset_index(name="n").pivot_table(
        index=["episode", "round_number"], columns="agent_group", values="n"
    )
    cpiv = cpiv.reindex(columns=[0, 1])
    return float(cpiv[group].isna().mean())


def matchup_episodes(df: pd.DataFrame, sides: dict, metric: str, agg: str):
    """One row per (matchup, episode) with both sides' scores."""
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
    return pd.DataFrame(rows)


def winrate_table(df: pd.DataFrame, sides: dict, metric: str, agg: str):
    """Per-matchup win-rate rows pooled over both position assignments."""
    res = matchup_episodes(df, sides, metric, agg)
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


def boot_mean(x: np.ndarray, rng) -> tuple:
    """95% bootstrap interval of the mean of x (resampling episodes)."""
    means = rng.choice(x, (N_BOOT, len(x))).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def boot_diff(x: np.ndarray, y: np.ndarray, rng) -> tuple:
    """95% bootstrap interval of mean(x) - mean(y), x and y independent."""
    dx = rng.choice(x, (N_BOOT, len(x))).mean(axis=1)
    dy = rng.choice(y, (N_BOOT, len(y))).mean(axis=1)
    return np.percentile(dx - dy, 2.5), np.percentile(dx - dy, 97.5)


def fmt_ci(m: float, lo: float, hi: float) -> str:
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def pool_h2h_table(df: pd.DataFrame, sides: dict) -> pd.DataFrame:
    """Definition 5: pool head to head, paired margin per matchup."""
    rng = np.random.default_rng(BOOT_SEED)
    res = matchup_episodes(df, sides, POOL, "sum")
    out = []
    for key in res["matchup"].unique():
        s = res[res["matchup"] == key]
        d = (s["a"] - s["b"]).to_numpy()
        lo, hi = boot_mean(d, rng)
        out.append(
            {
                "matchup (a vs b)": key,
                "episodes": len(s),
                "a_win%": round(100 * float((d > 0).mean()), 1),
                "a_pool": round(float(s["a"].mean()), 1),
                "b_pool": round(float(s["b"].mean()), 1),
                "margin a - b [95%]": fmt_ci(d.mean(), lo, hi),
            }
        )
    return pd.DataFrame(out)


def against_anchor(df: pd.DataFrame, anchor: str) -> dict:
    """Manager -> per-episode (pool, members) of its group against the anchor.

    Pools every pairing of the manager with the anchor, in either position.
    The anchor's own entry comes from `anchor_vs_anchor`, both groups.
    """
    scores = {}
    for run in df["run"].unique():
        g0, g1 = pairing_of(run)
        if anchor not in (g0, g1):
            continue
        sub = df[df["run"] == run]
        ep = group_round_scores(sub, POOL, "sum").groupby("episode").mean()
        for group, manager in ((0, g0), (1, g1)):
            other = g1 if group == 0 else g0
            if other != anchor:
                continue
            frame = pd.DataFrame(
                {"pool": ep[group], "members": episode_members(sub, group)}
            )
            scores.setdefault(manager, []).append(frame)
    return {m: pd.concat(f, ignore_index=True) for m, f in scores.items()}


def anchor_table(df: pd.DataFrame, anchor: str, reference: str):
    """Definition 4: rank managers by their group's pool against the anchor.

    Returns (table, warnings).
    """
    managers = sorted({m for run in df["run"].unique() for m in pairing_of(run)})
    scores = against_anchor(df, anchor)
    warnings = []
    if not scores:
        warnings.append(f"no manager played `{anchor}`: definition 4 skipped")
        return None, warnings
    missing = [m for m in managers if m not in scores and m != anchor]
    if missing:
        warnings.append(f"not compared, never played `{anchor}`: " + ", ".join(missing))
    ref = scores.get(reference)
    if ref is None:
        warnings.append(
            f"reference `{reference}` never played `{anchor}`: no margins; pass "
            f"the sim dir that has it (e.g. the run with {reference}_vs_{anchor})"
        )
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for m, s in sorted(scores.items(), key=lambda kv: -kv[1]["pool"].mean()):
        row = {
            "manager": m + (" (reference)" if m == reference else ""),
            "episodes": len(s),
            "pool": round(float(s["pool"].mean()), 1),
            "members": round(float(s["members"].mean()), 2),
        }
        if ref is not None:
            if m == reference:
                row[f"vs {reference} [95%]"] = "—"
            else:
                x, y = s["pool"].to_numpy(), ref["pool"].to_numpy()
                lo, hi = boot_diff(x, y, rng)
                row[f"vs {reference} [95%]"] = fmt_ci(x.mean() - y.mean(), lo, hi)
        out.append(row)
    return pd.DataFrame(out), warnings


def to_markdown(tbl: pd.DataFrame) -> str:
    cols = list(tbl.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in tbl.itertuples(index=False)
    ]
    return "\n".join([head, sep, *body])


def five_definitions(df, sides, sources, anchor, reference) -> str:
    blocks = [
        "# Win rates: the five definitions of #226",
        f"_Source: {sources} — {len(sides)} runs; head-to-head definitions "
        "pool both positions by matchup. Empty group-round = 0._",
    ]
    for title, metric, agg, score in HEAD_TO_HEAD:
        tbl = winrate_table(df, sides, metric, agg)
        blocks.append(
            f"\n## {title}\n\n{score}, averaged over the episode; the higher "
            f"group wins the episode.\n\n{to_markdown(tbl)}"
        )

    tbl, warnings = anchor_table(df, anchor, reference)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    text = (
        f"\n## 4. Levin's criterion: common pool against `{anchor}` (#219)\n\n"
        f"Per round, the group's common pool 1.6·Σc − Σp, averaged over the "
        f"episode. Managers never meet: each plays `{anchor}`, and **the one "
        f"whose group does best against `{anchor}` wins** (ranked below). "
        f"Margin = its group minus `{reference}`'s group against `{anchor}`, "
        "95% bootstrap interval over episodes; members = mean group size."
    )
    text += "".join(f"\n\n> ⚠ {w}" for w in warnings)
    if tbl is not None:
        text += f"\n\n{to_markdown(tbl)}"
    blocks.append(text)

    blocks.append(
        "\n## 5. Common pool, head to head\n\nThe same pool; the two groups of "
        "the same episode. Margin = a − b, paired by episode, 95% bootstrap "
        f"interval.\n\n{to_markdown(pool_h2h_table(df, sides))}"
    )
    return "\n".join(blocks)


def custom_tables(df, sides, sources, agg, metrics) -> str:
    blocks = [
        f"# Win rates ({agg} per group, empty round = 0)",
        f"_Source: {sources} — {len(sides)} runs pooled by matchup._",
    ]
    for metric in metrics:
        if metric not in df.columns:
            print(f"skipping {metric!r}: not a column", file=sys.stderr)
            continue
        tbl = winrate_table(df, sides, metric, agg)
        blocks.append(f"\n## {metric}\n\n{to_markdown(tbl)}")
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sim_dirs", nargs="+", help="Sim output dir(s) with per_round.parquet"
    )
    parser.add_argument(
        "--anchor",
        default="ah",
        help="Definition 4: the manager every policy plays (default ah)",
    )
    parser.add_argument(
        "--reference",
        default="zero",
        help="Definition 4: margins are taken against this manager (default zero)",
    )
    parser.add_argument(
        "--agg",
        choices=["sum", "mean"],
        default=None,
        help="Custom tables: per-group aggregation over agents each round",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=f"Custom tables: metrics to table (default: {' '.join(DEFAULT_METRICS)})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional markdown file to write the tables to",
    )
    args = parser.parse_args()

    df = load_per_round(args.sim_dirs)
    sides = pairing_sides(df)
    sources = ", ".join(os.path.join(d, "per_round.parquet") for d in args.sim_dirs)

    if args.agg is not None or args.metrics is not None:
        text = custom_tables(
            df, sides, sources, args.agg or "sum", args.metrics or DEFAULT_METRICS
        )
    else:
        text = five_definitions(df, sides, sources, args.anchor, args.reference)

    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
