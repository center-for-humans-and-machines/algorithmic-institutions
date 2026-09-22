"""Figures for the contributor punishment-targeting intervention.

Reads ``intervention_surface.csv``, ``level_slopes.csv`` and
``band_slopes.csv`` written by ``contributor_punishment_intervention.py`` and
draws, into the same directory:

  * ``intervention_surface.jpg`` -- the punishment effect, i.e. the forced
    response minus the same context's response at zero punishment, as curves
    (one per forced contribution level) and as a heatmap.  The raw response is
    dominated by mean reversion in the contribution itself, which would bury
    the dose response; subtracting the p = 0 column leaves exactly the effect
    of the punishment;
  * ``band_slopes.jpg``          -- the effect per punishment point against the
    contribution level, model against human, with the two noise scales drawn.

Colour: the contribution level is an ordered magnitude, so it gets a single
sequential ramp (light = gave little, dark = gave a lot); the signed effect
gets the diverging pair around a neutral zero.  The two encodings are kept
apart on purpose -- using one ramp for both would make "blue" mean a low
contribution in one panel and a negative effect in the other.

No torch, no torch_geometric: runs locally.
    python scripts/data_analysis/plot_contributor_intervention.py [--dir D]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "plots/data_analysis/contributor_punishment_targeting"
BAND_LABELS = ["0-4", "5-9", "10-14", "15-19"]
DIVERGING = "RdBu_r"
SEQUENTIAL = "mako_r"
INK = "#3d3d3d"


def _recess(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbdbd")
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)


def surface_figure(surf, out):
    piv = surf.pivot(index="contribution", columns="punishment", values="mean_dc")
    rel = piv.sub(piv[0], axis=0)  # the punishment effect proper
    levels = list(rel.index)
    cmap = sns.color_palette(SEQUENTIAL, as_cmap=True)
    # the ramp starts below the data so the lowest level is still legible ink
    norm = plt.Normalize(min(levels) - 5, max(levels))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    for c in levels:
        ax.plot(rel.columns, rel.loc[c], color=cmap(norm(c)), lw=1.6)
    ax.axhline(0, color=INK, lw=0.8, ls=":")
    for c in (0, 5, 12, 20):
        ax.annotate(
            f"c = {c}",
            xy=(rel.columns[-1], rel.loc[c].iloc[-1]),
            xytext=(-6, 4),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=cmap(norm(c)),
            fontweight="bold",
        )
    ax.set_xlabel("forced punishment at round t", color=INK)
    ax.set_ylabel(r"punishment effect   $\Delta$(c, p) $-$ $\Delta$(c, 0)", color=INK)
    ax.set_title(
        "What the punishment itself does\n"
        "compliance below, withdrawal above, sign flip near c = 12",
        color=INK,
        fontsize=11,
    )
    _recess(ax)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        label="forced contribution at round t",
    )
    cb.set_ticks([0, 5, 10, 15, 20])

    lim = float(np.abs(rel.to_numpy()).max())
    sns.heatmap(
        rel,
        ax=axes[1],
        cmap=DIVERGING,
        center=0,
        vmin=-lim,
        vmax=lim,
        cbar_kws={"label": "punishment effect (contribution points)"},
    )
    axes[1].invert_yaxis()
    axes[1].set_title("The intervention surface", color=INK, fontsize=11)
    axes[1].set_xlabel("forced punishment", color=INK)
    axes[1].set_ylabel("forced contribution", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def slope_figure(lev, band, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    ax = axes[0]
    ax.plot(
        lev["contribution"],
        lev["model_slope_uniform"],
        lw=1.4,
        color="#92c5de",
        label="model, forced, whole dose range 1-30",
    )
    ax.plot(
        lev["contribution"],
        lev["model_slope_human_weighted"],
        marker="o",
        ms=5,
        lw=2,
        color="#2166ac",
        label="model, forced, at the doses humans gave",
    )
    # the human slope one level at a time is too thin to read (n 16-360 per
    # level); its own banded statistic is the honest comparison
    spans = {
        "0-4": (-0.4, 4.4),
        "5-9": (4.6, 9.4),
        "10-14": (9.6, 14.4),
        "15-19": (14.6, 19.4),
    }
    hb = band.set_index("band")["human_slope"]
    for i, (lab, (x0, x1)) in enumerate(spans.items()):
        ax.plot(
            [x0, x1],
            [hb[lab]] * 2,
            lw=2.5,
            ls="--",
            color="#b2182b",
            label="human, observed, per band" if i == 0 else None,
        )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlabel("own contribution at round t", color=INK)
    ax.set_ylabel("contribution points per punishment point", color=INK)
    ax.set_title(
        "The targeting gradient at one-point resolution",
        color=INK,
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9)
    _recess(ax)

    ax = axes[1]
    band = band.set_index("band").loc[BAND_LABELS].reset_index()
    x = np.arange(len(BAND_LABELS))
    ax.bar(
        x - 0.2,
        band["human_slope"],
        width=0.38,
        color="#b2182b",
        label="human, observed",
    )
    err = np.vstack(
        [
            band["model_forced_slope"] - band["boot_lo"],
            band["boot_hi"] - band["model_forced_slope"],
        ]
    )
    ax.bar(
        x + 0.2,
        band["model_forced_slope"],
        width=0.38,
        yerr=err,
        capsize=3,
        ecolor=INK,
        color="#2166ac",
        label="model, forced (causal)",
    )
    for i, sd in enumerate(band["seed_sd"]):
        ax.plot(
            [x[i] + 0.2, x[i] + 0.2],
            [
                band["model_forced_slope"][i] - sd,
                band["model_forced_slope"][i] + sd,
            ],
            color="#7f7f7f",
            lw=4,
            alpha=0.5,
            solid_capstyle="butt",
        )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS)
    ax.set_xlabel("own contribution band", color=INK)
    ax.set_ylabel("RCE slope", color=INK)
    ax.set_title(
        "RCE and its interventional twin\n"
        "thin bars: context bootstrap 95% CI; grey: contributor seed sd",
        color=INK,
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9)
    _recess(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    d = Path(args.dir)
    surface_figure(
        pd.read_csv(d / "intervention_surface.csv"), d / "intervention_surface.jpg"
    )
    slope_figure(
        pd.read_csv(d / "level_slopes.csv"),
        pd.read_csv(d / "band_slopes.csv"),
        d / "band_slopes.jpg",
    )
    print(f"wrote {d / 'intervention_surface.jpg'}")
    print(f"wrote {d / 'band_slopes.jpg'}")


if __name__ == "__main__":
    main()
