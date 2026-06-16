"""Individual-trajectory heatmaps from a sim's per_round.parquet.

For one (pairing, episode) renders three vertically aligned agent x round
heatmaps (group membership, contribution, punishment) plus a per-group
payoff panel (the reward signal; --payoff-type sum|avg). Switch arrivals
are outlined on every panel; switch-decision rounds are dashed gridlines.
Cell annotations carry the values, so no colorbars are drawn.

Usage:
    python scripts/plotting/plot_sim_individual_heatmaps.py <sim_dir> \\
        [--pairing P ...] [--episode E] [--switch-every 4] \\
        [--payoff-type sum|avg] [--out-dir DIR]

Episode selection defaults to the episode with the most switch events in
each pairing. Examples:

    python scripts/plotting/plot_sim_individual_heatmaps.py \\
        plots/simulation/19_2g8a_rule_based_vs_zero \\
        --pairing rule_k1_vs_zero --episode 0
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

GROUP_CMAP = ["#4393c3", "#d6604d"]


def load_per_round(sim_dir: str) -> pd.DataFrame:
    path = os.path.join(sim_dir, "per_round.parquet")
    if not os.path.exists(path):
        sys.exit(f"per_round.parquet not found at {path}")
    df = pd.read_parquet(path)
    df["pairing"] = df["run"].str.replace(
        "ah group_switching managed by ", "", regex=False
    )
    df["pairing"] = df["pairing"].str.replace("ah managed by ", "", regex=False)
    df["agent"] = df["participant_code"].str.split("_").str[0].astype(int)
    df = df.sort_values(["pairing", "episode", "agent", "round_number"])
    diff = df.groupby(["pairing", "episode", "agent"])["group_id"].diff()
    df["switched"] = diff.ne(0) & diff.notna()
    return df


def plot_episode(
    e, pairing, episode, n_punishments, switch_every, out_path, payoff_type="sum"
):
    def piv(v):
        return e.pivot(index="agent", columns="round_number", values=v)

    n_rounds = e["round_number"].max() + 1
    panels = [
        (
            "group membership (blue=group 0, red=group 1)",
            piv("agent_group"),
            sns.color_palette(GROUP_CMAP, as_cmap=True),
            0,
            1,
            False,
        ),
        ("contribution (0-20)", piv("contribution"), "viridis", 0, 20, True),
        (
            f"punishment (0-{n_punishments - 1})",
            piv("punishment"),
            "rocket_r",
            0,
            n_punishments - 1,
            True,
        ),
    ]
    sw = piv("switched")
    ys, xs = np.where(sw.values)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 0.7]},
    )
    for ax, (title, mat, cmap, vmin, vmax, annot) in zip(axes[:3], panels):
        sns.heatmap(
            mat,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            annot=annot,
            fmt=".0f",
            annot_kws={"size": 7},
        )
        for y, x in zip(ys, xs):
            ax.add_patch(
                Rectangle(
                    (x, y), 1, 1, fill=False, edgecolor="magenta", lw=2.2, zorder=10
                )
            )
        if switch_every:
            for r in range(switch_every, n_rounds, switch_every):
                ax.axvline(r, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("agent")
        ax.set_xlabel("")

    # 4th panel: per-group payoff per round (the reward signal). Aggregated
    # over the agents in each group that round; group composition shifts as
    # agents switch. Lines centered on the heatmap cells (round + 0.5).
    pay, grp = piv("payoff"), piv("agent_group")
    rounds = pay.columns.values
    ax = axes[3]
    for g, color in ((0, GROUP_CMAP[0]), (1, GROUP_CMAP[1])):
        masked = pay.where(grp == g)
        series = masked.sum(axis=0) if payoff_type == "sum" else masked.mean(axis=0)
        ax.plot(
            rounds + 0.5,
            series.values,
            color=color,
            marker="o",
            ms=2.5,
            lw=1.8,
            label=f"group {g}",
        )
    if switch_every:
        for r in range(switch_every, n_rounds, switch_every):
            ax.axvline(r, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlim(0, n_rounds)
    ax.set_title(f"group payoff ({payoff_type} per round)", fontsize=10, loc="left")
    ax.set_ylabel("payoff")
    ax.set_xlabel("round")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Individual trajectories — {pairing}, episode {episode}\n"
        "(magenta border = switch arrival, dashed = switch-decision rounds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", help="Sim output dir with per_round.parquet")
    parser.add_argument(
        "--pairing",
        nargs="*",
        default=None,
        help="Pairings to plot (default: all in the parquet)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode to plot (default: episode with most switch events)",
    )
    parser.add_argument(
        "--switch-every",
        type=int,
        default=4,
        help="Rounds between switch decisions, for gridlines (default 4)",
    )
    parser.add_argument(
        "--n-punishments",
        type=int,
        default=31,
        help="Punishment level count, sets the color scale (default 31)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <sim_dir>)",
    )
    parser.add_argument(
        "--payoff-type",
        choices=["sum", "avg"],
        default="sum",
        help="Per-group payoff shown in the 4th panel (default sum)",
    )
    args = parser.parse_args()

    df = load_per_round(args.sim_dir)
    pairings = args.pairing or sorted(df["pairing"].unique())
    out_dir = args.out_dir or args.sim_dir
    os.makedirs(out_dir, exist_ok=True)

    for pairing in pairings:
        d = df[df["pairing"] == pairing]
        if d.empty:
            print(f"skipping {pairing!r}: not in parquet")
            continue
        if args.episode is not None:
            episode = args.episode
        else:
            episode = d.groupby("episode")["switched"].sum().idxmax()
        e = d[d["episode"] == episode]
        if e.empty:
            print(f"skipping {pairing!r} episode {episode}: no rows")
            continue
        out_path = os.path.join(
            out_dir, f"individual_heatmap_{pairing}_ep{episode}.jpg"
        )
        plot_episode(
            e,
            pairing,
            episode,
            args.n_punishments,
            args.switch_every,
            out_path,
            payoff_type=args.payoff_type,
        )


if __name__ == "__main__":
    main()
