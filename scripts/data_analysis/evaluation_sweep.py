"""Stack-sweep evaluation visuals (#139).

Aggregates the per-run evaluation scores of a self-play stack sweep into one
score matrix plus the slot-level story figures: the slot report card, the
ranking concordance, and the unrolled switch comparison. Scores are the #132
normalised scores (1 = human noise ceiling).

Every sim dir must contain evaluation/scores.csv and follow the 23-family
naming ..._self_{contr}_contr_{switch}_switch; the punisher axis comes from
the run names inside scores.csv.

Usage:
    python scripts/data_analysis/evaluation_sweep.py <output_name> \\
        <sim_dir> [<sim_dir> ...]

Outputs land in plots/data_analysis/evaluation/<output_name>/.
"""

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

METRIC_ORDER = [
    "CA",
    "CB",
    "CC",
    "CD",
    "CE",
    "CF",
    "CG",
    "SA",
    "SB",
    "SC",
    "PA",
    "PB",
    "PC",
    "PD",
    "RCA",
    "RCB",
    "RCC",
    "RCD",
    "RSA",
    "RPA",
    "RPB",
]

# metric family each slot owns (the rows of its report-card panel)
CONTR_ROWS = ["CA", "CB", "CC", "CD", "CE", "CF", "CG", "RCA", "RCB", "RCC", "RCD"]
SWITCH_ROWS = ["SA", "SB", "SC", "RSA"]
PUNISHER_ROWS = ["PA", "PB", "PC", "PD", "RPA", "RPB"]

CONTR_ORDER = ["gnn", "cat", "gaussian", "ridge"]
SWITCH_ORDER = ["lin", "gnn", "herdcopar1"]
PUNISHER_ORDER = ["multinomial", "multinomial_copula", "gnn", "gaussian", "ridge"]

SLOT_CONTEXT = {
    "contr": ["switch", "punisher"],
    "switch": ["contr", "punisher"],
    "punisher": ["contr", "switch"],
}

BAND_COLORS = ["#f7f7f7", "#fdd0a2", "#fc8d59", "#b30000"]
BAND_LABELS = [
    "<= 1  at the ceiling",
    "1-2  minor",
    "2-5  clear deviation",
    "> 5  not reproduced",
]
BAND_CMAP = ListedColormap(BAND_COLORS)
BAND_NORM = BoundaryNorm([0, 1, 2, 5, 100], BAND_CMAP.N)

PUNISHER_COLORS = {
    "multinomial": "#E69F00",
    "multinomial_copula": "#D55E00",
    "gnn": "#0072B2",
    "gaussian": "#CC79A7",
    "ridge": "#009E73",
}
CONTR_MARKERS = {"gnn": "o", "cat": "s", "gaussian": "^", "ridge": "D"}

DIR_PATTERN = re.compile(r"_self_(?P<contr>\w+?)_contr_(?P<switch>\w+?)_switch$")


def load_scores(sim_dirs):
    frames = []
    for d in sim_dirs:
        path = os.path.join(d, "evaluation", "scores.csv")
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found")
        match = DIR_PATTERN.search(os.path.basename(os.path.normpath(d)))
        if match is None:
            sys.exit(
                f"ERROR: cannot parse contr/switch from '{d}' "
                "(expected ..._self_<contr>_contr_<switch>_switch)"
            )
        df = pd.read_csv(path)
        df["contr"] = match["contr"]
        df["switch"] = match["switch"]
        frames.append(df)
    scores = pd.concat(frames, ignore_index=True)
    scores["punisher"] = (
        scores["run"]
        .str.replace("ah group_switching managed by ", "", regex=False)
        .str.replace("_self", "", regex=False)
        .str.replace("lin_", "", regex=False)
    )
    return scores


def write_matrix(scores, out_dir):
    matrix = scores.pivot_table(
        index=["contr", "switch", "punisher"], columns="metric", values="score"
    )[METRIC_ORDER].round(2)
    path = os.path.join(out_dir, "score_matrix.csv")
    matrix.to_csv(path)
    return path


def _band_legend(fig):
    handles = [
        mpatches.Patch(color=c, label=label)
        for c, label in zip(BAND_COLORS, BAND_LABELS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)


def _score_panel(ax, scores, slot, options, rows, title):
    m = scores.pivot_table(index="metric", columns=slot, values="score").reindex(rows)[
        options
    ]
    ax.imshow(m.values, cmap=BAND_CMAP, norm=BAND_NORM, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(options)):
            v = m.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 5 else "black",
            )
    ax.set_xticks(range(len(options)), options, fontsize=9)
    ax.set_yticks(range(len(rows)), rows, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)


def fig_slot_report(scores, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), width_ratios=[4, 2, 4])
    _score_panel(
        axes[0],
        scores,
        "contr",
        CONTR_ORDER,
        CONTR_ROWS,
        "contribution model\n(mean over switch x punisher)",
    )
    _score_panel(
        axes[1],
        scores,
        "switch",
        SWITCH_ORDER,
        SWITCH_ROWS,
        "switch model\n(mean over contr x punisher)",
    )
    _score_panel(
        axes[2],
        scores,
        "punisher",
        PUNISHER_ORDER,
        PUNISHER_ROWS,
        "punisher\n(mean over contr x switch)",
    )
    _band_legend(fig)
    fig.suptitle(
        "Score per slot, on the metric family the slot owns (lower = more human)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(out_dir, "slot_report.jpg"), dpi=150)
    plt.close(fig)


def _kendalls_w(scores, metric, slot):
    sub = scores[scores["metric"] == metric]
    wide = sub.pivot_table(index=SLOT_CONTEXT[slot], columns=slot, values="score")
    ranks = wide.rank(axis=1)
    n_contexts, n_options = wide.shape
    rank_sums = ranks.sum(axis=0)
    ss = ((rank_sums - rank_sums.mean()) ** 2).sum()
    w = 12 * ss / (n_contexts**2 * (n_options**3 - n_options))
    best = wide.mean().idxmin()
    best_wins = int((ranks[best] == 1.0).sum())
    return w, best, best_wins, n_contexts


def _concordance_panel(ax, scores, slot, rows, title):
    cmap = ListedColormap(["#f7f7f7", "#c6dbef", "#6baed6", "#08519c"])
    norm = BoundaryNorm([0.0, 0.1, 0.3, 0.5, 1.01], cmap.N)
    values, notes = [], []
    for metric in rows:
        w, best, wins, n_contexts = _kendalls_w(scores, metric, slot)
        values.append(w)
        notes.append(f"W={w:.2f}\n{best} best in {wins}/{n_contexts}")
    ax.imshow([[1.001 - v] for v in values], cmap=cmap, norm=norm, aspect="auto")
    for i, note in enumerate(notes):
        ax.text(
            0,
            i,
            note,
            ha="center",
            va="center",
            fontsize=8,
            color="white" if values[i] < 0.5 else "black",
        )
    ax.set_xticks([])
    ax.set_yticks(range(len(rows)), rows, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)


def fig_slot_concordance(scores, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), width_ratios=[4, 2, 4])
    _concordance_panel(
        axes[0],
        scores,
        "contr",
        CONTR_ROWS,
        "contribution model\n(ranking concordance across switch x punisher)",
    )
    _concordance_panel(
        axes[1],
        scores,
        "switch",
        SWITCH_ROWS,
        "switch model\n(concordance across contr x punisher)",
    )
    _concordance_panel(
        axes[2],
        scores,
        "punisher",
        PUNISHER_ROWS,
        "punisher\n(concordance across contr x switch)",
    )
    handles = [
        mpatches.Patch(color=c, label=label)
        for c, label in zip(
            ["#f7f7f7", "#c6dbef", "#6baed6", "#08519c"],
            [
                "W >= 0.9  identical ordering",
                "0.7-0.9",
                "0.5-0.7",
                "< 0.5  ordering scrambles",
            ],
        )
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "Kendall's W: does the option ranking survive across the marginalised "
        "contexts?",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(out_dir, "slot_concordance.jpg"), dpi=150)
    plt.close(fig)


def fig_switch_unrolled(scores, out_dir):
    fig, axes = plt.subplots(1, len(SWITCH_ROWS), figsize=(14, 4.2))

    hi = 6
    lo = 0

    for ax, metric in zip(axes, SWITCH_ROWS):
        sub = scores[scores["metric"] == metric]
        wide = sub.pivot_table(
            index=["contr", "punisher"], columns="switch", values="score"
        )
        ax.plot([lo, hi], [lo, hi], color="gray", linewidth=0.8, zorder=1)
        for (contr, punisher), row in wide.iterrows():
            ax.scatter(
                row["lin"],
                row["gnn"],
                color=PUNISHER_COLORS[punisher],
                marker=CONTR_MARKERS[contr],
                s=45,
                zorder=2,
                edgecolors="white",
                linewidths=0.5,
            )
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("score with lin switch")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, linewidth=0.5)
        for b in [1,2,5]:
            ax.vlines(x=b, ymin=lo, ymax=hi, colors='k', alpha=0.5, linestyles='dashed')
            ax.hlines(y=b, xmin=lo, xmax=hi, colors='k', alpha=0.5, linestyles='dashed')
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("score with gnn switch")
    handles = [
        Line2D([], [], marker="o", ls="", color=c, label=f"{p} punisher")
        for p, c in PUNISHER_COLORS.items()
    ]
    handles += [
        Line2D([], [], marker=m, ls="", color="#555555", label=f"{c} contr")
        for c, m in CONTR_MARKERS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=8.5)
    fig.suptitle(
        "Switch slot unrolled: every context, lin vs gnn switch "
        "(below the diagonal = gnn switch better)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(out_dir, "switch_unrolled.jpg"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_name",
        help="Folder name under plots/data_analysis/evaluation/ for the outputs",
    )
    parser.add_argument(
        "sim_dirs", nargs="+", help="Sim output dirs with evaluation/scores.csv"
    )
    args = parser.parse_args()

    args.out = os.path.join("plots/data_analysis/evaluation", args.output_name)
    os.makedirs(args.out, exist_ok=True)
    scores = load_scores(args.sim_dirs)
    matrix_path = write_matrix(scores, args.out)
    fig_slot_report(scores, args.out)
    fig_slot_concordance(scores, args.out)
    fig_switch_unrolled(scores, args.out)
    n_runs = len(scores[["contr", "switch", "punisher"]].drop_duplicates())
    print(f"{n_runs} runs -> {matrix_path} + 3 figures in {args.out}")


if __name__ == "__main__":
    main()
