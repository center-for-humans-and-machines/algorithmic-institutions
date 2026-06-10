"""Plot opposition-group contribution and switch-in rate per policy.

For each pair of managers (X, opposition) that appears as both
`X_vs_<opposition>` and `<opposition>_vs_X`, computes the opposition group's
behavior averaged across the two mirror pairings: contribution per round and
switch-in rate per round. Emits one chart with two side-by-side line plots,
one line per X (the policy of interest).

Usage:
    python scripts/plotting/plot_sim_opposition_behavior.py <sim_dir> \\
        [--opposition zero_punishment] [--managers M1 M2 ...] [--out FILE] \\
        [--n-agents 8]

Examples:
    python scripts/plotting/plot_sim_opposition_behavior.py \\
        plots/simulation/19_2g8a_rule_based_vs_zero \\
        --opposition zero --managers rule_k1 rule_k4 rule_k8
"""
import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_per_round(sim_dir: str) -> pd.DataFrame:
    path = os.path.join(sim_dir, "per_round.parquet")
    if not os.path.exists(path):
        sys.exit(f"per_round.parquet not found at {path}")
    df = pd.read_parquet(path)
    df["pairing"] = df["run"].str.replace(
        "ah group_switching managed by ", "", regex=False
    )
    df["pairing"] = df["pairing"].str.replace("ah managed by ", "", regex=False)
    return df


def find_mirror_pairs(pairings, opposition):
    # Returns {policy: [(pairing_name, policy_group_id), ...]} for policies
    # that appear in both `policy_vs_opposition` and `opposition_vs_policy`.
    forward, backward = {}, {}
    pat = re.compile(r"^(.+)_vs_(.+)$")
    for p in pairings:
        m = pat.match(p)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        if b == opposition and a != opposition:
            forward[a] = p
        elif a == opposition and b != opposition:
            backward[b] = p
    out = {}
    for policy in set(forward) & set(backward):
        out[policy] = [(forward[policy], 1), (backward[policy], 0)]
    return out


def compute_metrics(df, mirror_pairs, n_agents):
    cond_map, opp_g_map = {}, {}
    for policy, entries in mirror_pairs.items():
        for pairing, opp_g in entries:
            cond_map[pairing] = policy
            opp_g_map[pairing] = opp_g
    df = df[df["pairing"].isin(cond_map)].copy()
    df["condition"] = df["pairing"].map(cond_map)
    df["opp_g"] = df["pairing"].map(opp_g_map)
    df["is_opp"] = df["group_id"] == df["opp_g"]

    df = df.sort_values(
        ["pairing", "episode", "participant_code", "round_number"]
    ).copy()
    df["prev_g"] = df.groupby(["pairing", "episode", "participant_code"])[
        "group_id"
    ].shift(1)
    df["switched_in_to_opp"] = (
        (df["group_id"] == df["opp_g"])
        & (df["prev_g"] != df["opp_g"])
        & df["prev_g"].notna()
    )

    contr = (
        df[df["is_opp"]]
        .groupby(["condition", "episode", "round_number"])["contribution"]
        .mean()
        .reset_index()
    )
    switches = (
        df.groupby(["condition", "pairing", "episode", "round_number"])[
            "switched_in_to_opp"
        ]
        .sum()
        .reset_index()
        .rename(columns={"switched_in_to_opp": "n_switched_in"})
    )
    switches["switch_rate"] = switches["n_switched_in"] / n_agents
    return contr, switches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", help="Sim output directory with per_round.parquet")
    parser.add_argument(
        "--opposition",
        default="zero_punishment",
        help="Opposition manager name (default zero_punishment). Falls back to "
        "'zero' if no pairings match.",
    )
    parser.add_argument(
        "--managers",
        nargs="*",
        default=None,
        help="Restrict to these policies (default: all that mirror against opposition)",
    )
    parser.add_argument(
        "--n-agents", type=int, default=8, help="Total agents per run (default 8)"
    )
    parser.add_argument("--out", default=None, help="Output image path")
    args = parser.parse_args()

    df = load_per_round(args.sim_dir)
    pairings = df["pairing"].unique().tolist()
    mp = find_mirror_pairs(pairings, args.opposition)
    if not mp and args.opposition == "zero_punishment":
        mp = find_mirror_pairs(pairings, "zero")
    if not mp:
        sys.exit(
            f"no mirror pairs found around opposition={args.opposition!r}. "
            f"available pairings: {pairings}"
        )
    if args.managers:
        mp = {k: v for k, v in mp.items() if k in args.managers}
    if not mp:
        sys.exit("no policies left after --managers filter")

    contr, switches = compute_metrics(df, mp, args.n_agents)

    policies = sorted(mp.keys())
    cmap = plt.get_cmap("tab10")
    palette = {p: cmap(i % 10) for i, p in enumerate(policies)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(
        data=contr,
        x="round_number",
        y="contribution",
        hue="condition",
        ax=axes[0],
        palette=palette,
        errorbar=("ci", 95),
        linewidth=2,
        hue_order=policies,
    )
    axes[0].set_title(
        "Opposition-group contribution\n(mirror-averaged across 2 pairings per policy)"
    )
    axes[0].set_ylabel("contribution (avg per agent)")
    axes[0].set_xlabel("round_number")

    sns.lineplot(
        data=switches,
        x="round_number",
        y="switch_rate",
        hue="condition",
        ax=axes[1],
        palette=palette,
        errorbar=("ci", 95),
        linewidth=2,
        hue_order=policies,
    )
    axes[1].set_title(
        f"Switch-in rate to opposition\n(# joining opposition / {args.n_agents} per round)"
    )
    axes[1].set_ylabel("switch-in rate")
    axes[1].set_xlabel("round_number")

    fig.tight_layout()
    out = args.out or os.path.join(args.sim_dir, "opposition_behavior.jpg")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
