"""Plot per-pairing variable trends and group_size from a simulation
per_round.parquet. One script produces both plot types for a chosen pairing
(or all pairings).

Usage:
    python scripts/plotting/plot_sim_pairings.py <sim_dir> [--pairings P1 P2 ...] [--out-dir DIR]

Examples:
    python scripts/plotting/plot_sim_pairings.py plots/simulation/19_2g8a_rule_based_vs_zero
    python scripts/plotting/plot_sim_pairings.py plots/simulation/19_2g8a_rule_based_vs_zero \\
        --pairings rule_k1_vs_zero zero_vs_rule_k1

For each chosen pairing, emits:
    <out_dir>/pairing_<name>.jpg          5-panel variable trend (punishment,
                                          contribution, common_good, payoff,
                                          payoff_sum) per side
    <out_dir>/group_size_<name>.jpg       group_size per side with empty-group
                                          rows back-filled so means sum to
                                          n_agents at every round
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


VARIABLES = ["punishment", "contribution", "common_good", "payoff", "payoff_sum"]


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


def attach_payoff_sum(df: pd.DataFrame) -> pd.DataFrame:
    key = ["pairing", "episode", "round_number", "group_id"]
    df = df.copy()
    df["payoff_sum"] = (
        df.groupby(key)["payoff"].transform("sum").where(~df.duplicated(key))
    )
    return df


def compute_group_size(df: pd.DataFrame, n_groups: int = 2) -> pd.DataFrame:
    # Counts of agents per (pairing, episode, round, group_id); back-fill
    # zero-sized groups so per-round means across episodes sum to n_agents.
    sizes = (
        df.groupby(["pairing", "episode", "round_number", "group_id"])
        .size()
        .unstack("group_id", fill_value=0)
    )
    for g in range(n_groups):
        if g not in sizes.columns:
            sizes[g] = 0
    sizes = sizes[sorted(sizes.columns)]
    return sizes.stack(future_stack=True).reset_index(name="group_size")


def parse_sides(pairing: str) -> tuple:
    parts = pairing.split("_vs_")
    if len(parts) == 2:
        return parts[0], parts[1]
    return f"g0:{pairing}", f"g1:{pairing}"


def plot_pairing_variables(
    df_pair: pd.DataFrame, pairing: str, out_path: str
) -> None:
    m0, m1 = parse_sides(pairing)
    sub = df_pair.copy()
    sub["side"] = sub["group_id"].map({0: f"g0: {m0}", 1: f"g1: {m1}"})
    palette = {f"g0: {m0}": "C0", f"g1: {m1}": "C3"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, var in zip(axes, VARIABLES):
        sns.lineplot(
            data=sub,
            x="round_number",
            y=var,
            hue="side",
            ax=ax,
            palette=palette,
            errorbar=("ci", 95),
            linewidth=1.8,
        )
        ax.set_title(var)
        ax.set_xlabel("round_number")
        ax.set_ylabel("")
        if ax is not axes[0]:
            leg = ax.get_legend()
            if leg:
                leg.remove()
    axes[-1].axis("off")
    fig.suptitle(
        f"pairing: {pairing}  (per_round.parquet, 95% CI across episodes)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_pairing_group_size(
    gs_pair: pd.DataFrame, pairing: str, out_path: str, n_agents: int
) -> None:
    m0, m1 = parse_sides(pairing)
    sub = gs_pair.copy()
    sub["side"] = sub["group_id"].map({0: f"g0: {m0}", 1: f"g1: {m1}"})
    palette = {f"g0: {m0}": "C0", f"g1: {m1}": "C3"}
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=sub,
        x="round_number",
        y="group_size",
        hue="side",
        ax=ax,
        palette=palette,
        errorbar=("ci", 95),
        linewidth=2,
    )
    ax.axhline(
        n_agents / 2, ls=":", color="gray", alpha=0.5, label=f"start ({n_agents // 2})"
    )
    ax.axhline(
        n_agents, ls=":", color="black", alpha=0.3, label=f"max ({n_agents})"
    )
    ax.set_ylim(0, n_agents + 0.5)
    ax.set_title(
        f"group_size: {pairing}  (95% CI across episodes, empty groups = 0)"
    )
    ax.set_xlabel("round_number")
    ax.set_ylabel("group_size (# agents)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", help="Sim output directory with per_round.parquet")
    parser.add_argument(
        "--pairings",
        nargs="*",
        default=None,
        help="Subset of pairings to plot (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots (default: same as sim_dir)",
    )
    parser.add_argument(
        "--n-groups", type=int, default=2, help="Number of groups (default 2)"
    )
    parser.add_argument(
        "--n-agents", type=int, default=8, help="Total agents per run (default 8)"
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.sim_dir
    os.makedirs(out_dir, exist_ok=True)

    df = load_per_round(args.sim_dir)
    df = attach_payoff_sum(df)
    gs = compute_group_size(df, n_groups=args.n_groups)

    pairings = args.pairings or sorted(df["pairing"].unique())
    for p in pairings:
        if p not in df["pairing"].unique():
            print(f"skip {p!r} — not found in {args.sim_dir}")
            continue
        var_path = os.path.join(out_dir, f"pairing_{p}.jpg")
        gs_path = os.path.join(out_dir, f"group_size_{p}.jpg")
        plot_pairing_variables(df[df["pairing"] == p], p, var_path)
        plot_pairing_group_size(gs[gs["pairing"] == p], p, gs_path, args.n_agents)
        print(f"saved {var_path}")
        print(f"saved {gs_path}")


if __name__ == "__main__":
    main()
