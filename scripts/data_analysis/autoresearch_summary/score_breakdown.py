"""Score breakdown along the two success spines: all 21 rows as separate
lines colored by the stack slot they measure (contribution / switch /
punisher), with the 21-row mean overlaid bold. Reads the cache written by
score_progressions.py.

Usage:
    python scripts/data_analysis/autoresearch_summary/score_breakdown.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from score_progressions import (
    DATA, FAMILY, FAMILY_COLOR, METRICS, SURFACE, INK, INK_2,
    load_spine_scores, spines,
)

OUT = Path("plots/data_analysis/autoresearch_summary/score_breakdown.png")

LABEL_THRESHOLD = 2.0  # direct-label rows that ever exceed this


def main():
    experiments = json.loads((DATA / "experiments.json").read_text())
    trees = spines(experiments)
    all_prs = [pr for chain in trees.values() for pr in chain]
    root, cache = load_spine_scores(experiments, all_prs)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 7), facecolor=SURFACE, sharey=True,
        gridspec_kw={"width_ratios": [len(c) for c in trees.values()]},
    )
    for ax, (tree, chain) in zip(axes, trees.items()):
        ax.set_facecolor(SURFACE)
        steps = range(len(chain) + 1)
        for band in (1.0, 2.0, 5.0):
            ax.axhline(band, color="#eceae6", lw=0.8, zorder=1)
        labels = []
        for metric in METRICS:
            ys = [root[metric]] + [cache[str(pr)][metric] for pr in chain]
            color = FAMILY_COLOR[FAMILY[metric]]
            ax.plot(steps, ys, color=color, lw=1.2, alpha=0.65, zorder=3,
                    marker="o", markersize=2.5)
            if max(ys) >= LABEL_THRESHOLD:
                labels.append([metric, ys[-1], color])
        # dodge end labels apart in log space (min ratio between neighbours)
        labels.sort(key=lambda item: item[1])
        for i in range(1, len(labels)):
            labels[i][1] = max(labels[i][1], labels[i - 1][1] * 1.09)
        for metric, y, color in labels:
            ax.annotate(metric, (len(chain), y), fontsize=8, color=color,
                        xytext=(6, 0), textcoords="offset points",
                        va="center")
        means = [sum(root[m] for m in METRICS) / len(METRICS)] + [
            sum(cache[str(pr)][m] for m in METRICS) / len(METRICS)
            for pr in chain
        ]
        ax.plot(steps, means, color=INK, lw=2.6, zorder=4, marker="o",
                markersize=5, markeredgecolor=SURFACE)
        ax.annotate("mean", (len(chain), means[-1]), fontsize=9,
                    color=INK, fontweight="bold", xytext=(5, 0),
                    textcoords="offset points", va="center")
        ax.set_yscale("log")
        ax.set_yticks([0.5, 1, 2, 5, 10])
        ax.set_yticklabels(["0.5", "1", "2", "5", "10"])
        ax.set_xticks(list(steps))
        ax.set_xticklabels(["main"] + [f"#{pr}" for pr in chain], fontsize=8.5)
        ax.set_xlim(-0.2, len(chain) + 0.75)
        ax.set_title(f"{tree} spine", fontsize=11, color=INK)
        ax.tick_params(colors=INK_2, labelsize=8.5)
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d6d5d0")

    axes[0].set_ylabel("score (log; guides at band edges 1 / 2 / 5)",
                       color=INK_2, fontsize=9.5)
    fig.legend(handles=[
        *(Line2D([], [], color=c, lw=2, label=f"{f} rows")
          for f, c in FAMILY_COLOR.items()),
        Line2D([], [], color=INK, lw=2.6, label="21-row mean"),
    ], loc="lower center", ncol=4, frameon=False, fontsize=9.5)
    fig.suptitle(
        "All 21 scores along each success spine, colored by the stack slot "
        "they measure", fontsize=12, color=INK,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
