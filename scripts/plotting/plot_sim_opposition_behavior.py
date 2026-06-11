"""Plot opposition-group contribution and switch-in rate per policy.

For each pair of managers (X, opposition) that appears as both
`X_vs_<opposition>` and `<opposition>_vs_X`, computes the opposition group's
behavior averaged across the two mirror pairings: contribution per round and
switch-in rate per round. Emits two charts:

1. opposition_behavior.jpg: 2-panel line plot — opposition contribution and
   switch-in rate vs round.
2. opposition_first_switch_breakdown.jpg: per-policy faceted line plot of
   opposition-group contribution split into stayer / leaver / joiner roles
   around the first switch. Each role's curve only spans the rounds where
   the agent is actually in the opposition group.

Usage:
    python scripts/plotting/plot_sim_opposition_behavior.py <sim_dir> \\
        [--opposition zero_punishment] [--managers M1 M2 ...] [--out FILE] \\
        [--breakdown-out FILE] [--switch-round 4] [--n-rounds-shown 8] \\
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


def first_switch_role_frame(
    df, mirror_pairs, switch_round: int, n_rounds_shown: int
):
    # For each (pairing, episode, participant) classify by group membership
    # at switch_round-1 (pre) and switch_round (post) relative to that
    # pairing's opposition group, and return the full trajectory plus an
    # is_opp flag so downstream plotting can render out-of-opposition
    # segments separately as reference lines.
    cond_map, opp_g_map = {}, {}
    for policy, entries in mirror_pairs.items():
        for pairing, opp_g in entries:
            cond_map[pairing] = policy
            opp_g_map[pairing] = opp_g
    df = df[df["pairing"].isin(cond_map)].copy()
    df["condition"] = df["pairing"].map(cond_map)
    df["opp_g"] = df["pairing"].map(opp_g_map)
    df["is_opp"] = df["group_id"] == df["opp_g"]
    df = df[df["round_number"] < n_rounds_shown]

    key = ["pairing", "episode", "participant_code"]
    pre = (
        df[df["round_number"] == switch_round - 1]
        .set_index(key)["group_id"]
        .rename("pre_g")
    )
    post = (
        df[df["round_number"] == switch_round]
        .set_index(key)["group_id"]
        .rename("post_g")
    )
    roles = pd.concat([pre, post], axis=1).reset_index()
    opp_lookup = (
        df[key + ["opp_g", "condition"]].drop_duplicates(subset=key)
    )
    roles = roles.merge(opp_lookup, on=key)
    roles = roles.dropna(subset=["pre_g", "post_g"])
    roles["role"] = "other"
    in_pre = roles["pre_g"] == roles["opp_g"]
    in_post = roles["post_g"] == roles["opp_g"]
    roles.loc[in_pre & in_post, "role"] = "stayer"
    roles.loc[in_pre & ~in_post, "role"] = "leaver"
    roles.loc[~in_pre & in_post, "role"] = "joiner"
    roles = roles[roles["role"] != "other"]

    return df.merge(roles[key + ["role"]], on=key)


def plot_first_switch_breakdown(
    df,
    mirror_pairs,
    out_path: str,
    switch_round: int = 4,
    n_rounds_shown: int = 8,
):
    frame = first_switch_role_frame(
        df, mirror_pairs, switch_round, n_rounds_shown
    )
    in_opp = frame[frame["is_opp"]]
    leaver_after = frame[(frame["role"] == "leaver") & (~frame["is_opp"])]
    joiner_before = frame[(frame["role"] == "joiner") & (~frame["is_opp"])]
    policies = sorted(mirror_pairs.keys())
    role_palette = {"stayer": "C0", "leaver": "C3", "joiner": "C2"}
    # Desaturated tints for the out-of-opposition reference segments —
    # keeps the role identity visible while signalling "reference / not
    # in opposition this round".
    leaver_ref_color = sns.desaturate(role_palette["leaver"], 0.25)
    joiner_ref_color = sns.desaturate(role_palette["joiner"], 0.25)
    n = len(policies)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, policy in zip(axes, policies):
        sns.lineplot(
            data=in_opp[in_opp["condition"] == policy],
            x="round_number",
            y="contribution",
            hue="role",
            ax=ax,
            palette=role_palette,
            errorbar=("ci", 95),
            linewidth=2,
            hue_order=["stayer", "leaver", "joiner"],
            legend=False,
        )
        for ref, color in (
            (leaver_after, leaver_ref_color),
            (joiner_before, joiner_ref_color),
        ):
            sub_ref = ref[ref["condition"] == policy]
            if not sub_ref.empty:
                sns.lineplot(
                    data=sub_ref,
                    x="round_number",
                    y="contribution",
                    ax=ax,
                    color=color,
                    errorbar=("ci", 95),
                    linewidth=1.5,
                    linestyle="--",
                    legend=False,
                )
        ax.axvline(switch_round - 0.5, ls="--", color="gray", alpha=0.6)
        ax.set_title(policy)
        ax.set_xlabel("round_number")
        ax.set_xticks(range(n_rounds_shown))
    axes[0].set_ylabel("contribution (avg per agent)")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=role_palette["stayer"], lw=2, label="stayer"),
        Line2D([0], [0], color=role_palette["leaver"], lw=2, label="leaver (in opposition)"),
        Line2D([0], [0], color=role_palette["joiner"], lw=2, label="joiner (in opposition)"),
        Line2D([0], [0], color=leaver_ref_color, lw=1.5, ls="--",
               label="leaver (post-leave, in rule group)"),
        Line2D([0], [0], color=joiner_ref_color, lw=1.5, ls="--",
               label="joiner (pre-join, in rule group)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Opposition-group contribution by role around the first switch "
        f"(round {switch_round}); solid = while in opposition, "
        "dashed tinted = same agents while in the rule group",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument(
        "--breakdown-out",
        default=None,
        help="Output path for the first-switch role breakdown plot "
        "(default: <sim_dir>/opposition_first_switch_breakdown.jpg)",
    )
    parser.add_argument(
        "--switch-round",
        type=int,
        default=4,
        help="Round at which the first switch occurs (default 4)",
    )
    parser.add_argument(
        "--n-rounds-shown",
        type=int,
        default=8,
        help="Number of rounds to include in the breakdown plot (default 8)",
    )
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

    breakdown_out = args.breakdown_out or os.path.join(
        args.sim_dir, "opposition_first_switch_breakdown.jpg"
    )
    plot_first_switch_breakdown(
        df,
        mp,
        breakdown_out,
        switch_round=args.switch_round,
        n_rounds_shown=args.n_rounds_shown,
    )
    print(f"saved {breakdown_out}")


if __name__ == "__main__":
    main()
