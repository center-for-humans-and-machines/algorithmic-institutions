"""Visualisations for the evaluation suite (#137).

One figure per metric-table row, overlaying the human reference and
every simulation pairing -- the same object the row's metric measures.
Histograms show probability (each source's bars sum to 1), never raw
counts, because sources differ in episode count. Files land in
<output_dir>/evaluation/visuals/<name>.jpg.

Colors: the human reference is always black with a heavier line (so it
is identifiable beyond color alone); pairings take the Okabe-Ito
colorblind-safe hues in fixed order, never cycled -- more pairings than
hues is an error, not a generated color.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HUMAN_COLOR = "#000000"
PAIRING_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2"]

PLOTS = []


def plot(name):
    """Register a figure under its visuals/<name>.jpg file name."""

    def register(fn):
        PLOTS.append((name, fn))
        return fn

    return register


def series(sims):
    """(label, frame, color) per pairing, fixed order by run name."""
    runs = sorted(sims)
    if len(runs) > len(PAIRING_COLORS):
        raise ValueError(
            f"{len(runs)} pairings but only {len(PAIRING_COLORS)} fixed "
            "hues -- facet or fold instead of cycling colors"
        )
    return [
        (run.replace("ah group_switching managed by ", ""), sims[run], color)
        for run, color in zip(runs, PAIRING_COLORS)
    ]


def prob_hist(ax, human, sims, values, bins, xlabel, log_y=False):
    """Overlaid step histograms of `values(frame)`, each normalised to
    probability."""
    for label, df, color, lw in _sources(human, sims):
        v = values(df)
        ax.hist(
            v,
            bins=bins,
            weights=[1 / len(v)] * len(v),
            histtype="step",
            color=color,
            linewidth=lw,
            label=label,
        )
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability")


def lineplot(ax, human, sims, stat, xlabel, ylabel):
    """Overlaid lines of `stat(frame)` (a pd.Series, index -> value)."""
    for label, df, color, lw in _sources(human, sims):
        s = stat(df)
        ax.plot(s.index, s.values, color=color, linewidth=lw, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _sources(human, sims):
    """Human first (black, heavier), then the pairings in fixed order."""
    out = [("human", human, HUMAN_COLOR, 2.2)]
    out += [(label, df, color, 1.4) for label, df, color in series(sims)]
    return out


def plot_all(human, sims, out_dir):
    """Render every registered figure; returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, fn in PLOTS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fn(ax, human, sims)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{name}.jpg")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
