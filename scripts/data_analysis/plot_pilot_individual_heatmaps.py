"""Individual-trajectory heatmaps from the pilot group-switching CSV.

Pilot-data twin of scripts/plotting/plot_sim_individual_heatmaps.py: for
one episode renders three vertically aligned agent x round heatmaps
(group membership, contribution, punishment) plus a per-group payoff
panel (--payoff-type sum|avg). Switch arrivals are outlined on every
panel; switch-decision rounds are dashed gridlines. Cells where the
player gave no input (contribution) or the manager gave no input
(punishment) are greyed out instead of showing imputed values.

The doubled dataset contains each competition twice (original and
"(flipped)" label mapping); flipped copies are skipped by default since
they are identical up to group-color inversion.

Usage:
    python scripts/data_analysis/plot_pilot_individual_heatmaps.py \\
        experiments/2group_8agent_50ep.csv \\
        [--episode GLOBAL_GROUP_ID ...] [--switch-every 4] \\
        [--include-flipped] [--out-dir DIR]

Episode selection defaults to the episode with the most switch events.
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

GROUP_CMAP = ["#4393c3", "#d6604d"]
INVALID_COLOR = "#cccccc"


def load_pilot(csv_path: str, include_flipped: bool) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        sys.exit(f"pilot CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df["experiment_name"] == "ah_group_switching"].copy()
    if not include_flipped:
        df = df[~df["global_group_id"].str.contains(r"\(flipped\)", regex=True)]
    df = df.sort_values(["global_group_id", "player_id", "round_number"])
    diff = df.groupby(["global_group_id", "player_id"])["group_id"].diff()
    df["switched"] = diff.ne(0) & diff.notna()
    return df


def plot_episode(e, episode, n_punishments, switch_every, out_path, payoff_type="sum"):
    def piv(v):
        return e.pivot(index="player_id", columns="round_number", values=v)

    n_rounds = e["round_number"].max() + 1
    no_input_player = piv("player_no_input").astype(bool)
    no_input_manager = piv("manager_no_input").astype(bool)
    panels = [
        (
            "group membership (blue=group 0, red=group 1)",
            piv("group_id"),
            sns.color_palette(GROUP_CMAP, as_cmap=True),
            0,
            1,
            False,
            None,
        ),
        (
            "contribution (0-20; grey = player no input)",
            piv("contribution"),
            "viridis",
            0,
            20,
            True,
            no_input_player,
        ),
        (
            f"punishment (0-{n_punishments - 1}; grey = manager no input)",
            piv("punishment"),
            "rocket_r",
            0,
            n_punishments - 1,
            True,
            no_input_manager,
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
    for ax, (title, mat, cmap, vmin, vmax, annot, mask) in zip(axes[:3], panels):
        ax.set_facecolor(INVALID_COLOR)
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
            mask=mask,
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
        ax.set_ylabel("player")
        ax.set_xlabel("")

    # 4th panel: per-group payoff per round (reward signal), aggregated over
    # the players in each group that round. Lines centered on heatmap cells.
    pay, grp = piv("payoff"), piv("group_id")
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
        f"Pilot individual trajectories — {episode}\n"
        "(magenta border = switch arrival, dashed = switch-decision rounds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Pilot CSV (e.g. experiments/...50ep.csv)")
    parser.add_argument(
        "--episode",
        nargs="*",
        default=None,
        help="global_group_id(s) to plot (default: episode with most switches)",
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
        "--include-flipped",
        action="store_true",
        help="Also allow '(flipped)' duplicate episodes",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/pilot_data",
        help="Output directory (default plots/pilot_data)",
    )
    parser.add_argument(
        "--payoff-type",
        choices=["sum", "avg"],
        default="sum",
        help="Per-group payoff shown in the 4th panel (default sum)",
    )
    args = parser.parse_args()

    df = load_pilot(args.csv_path, args.include_flipped)
    if args.episode:
        episodes = args.episode
    else:
        episodes = [df.groupby("global_group_id")["switched"].sum().idxmax()]
    os.makedirs(args.out_dir, exist_ok=True)

    for episode in episodes:
        e = df[df["global_group_id"] == episode]
        if e.empty:
            print(f"skipping {episode!r}: not found")
            continue
        slug = re.sub(r"\W+", "_", episode).strip("_")
        out_path = os.path.join(args.out_dir, f"individual_heatmap_{slug}.jpg")
        plot_episode(
            e,
            episode,
            args.n_punishments,
            args.switch_every,
            out_path,
            payoff_type=args.payoff_type,
        )


if __name__ == "__main__":
    main()
