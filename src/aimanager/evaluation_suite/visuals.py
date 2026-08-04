"""Visualisations for the evaluation suite (#137).

One figure per metric-table row, overlaying the human reference and
every simulation pairing -- the same object the row's metric measures.
Histograms show probability (each source's bars sum to 1), never raw
counts, because sources differ in episode count. Files land in
<output_dir>/evaluation/visuals/<name>.jpg.

Colors: the human reference is always black with a heavier line (so it
is identifiable beyond color alone); pairings take the Okabe-Ito
colorblind-safe hues in fixed order, never cycled -- runs with more
pairings than curated hues get a deterministic distinctipy extension.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from aimanager.evaluation_suite.metrics import (  # noqa: E402
    RCB_LABELS,
    RPA_LABELS,
    RPB_LABELS,
    RSA_LABELS,
    ContributionMetrics,
    PunishmentMetrics,
    ResponseMetrics,
    SwitchingMetrics,
)

# Okabe-Ito colorblind-safe hues, ordered so consecutive assignments are
# maximally separated; pairwise distinctness (incl. against the black
# human reference) is enforced by test_palette_pairwise_distinct. Runs
# with more pairings than curated hues get a deterministic distinctipy
# extension instead of cycling; black stays reserved for the human.
HUMAN_COLOR = "#000000"
PAIRING_COLORS = ["#E69F00", "#0072B2", "#CC79A7", "#009E73", "#56B4E9"]


def pairing_colors(n):
    """n distinct hex colors: the curated prefix, extended by
    distinctipy (seeded, excluding the prefix, black and white) when a
    run has more pairings than curated hues."""
    if n <= len(PAIRING_COLORS):
        return PAIRING_COLORS[:n]
    import distinctipy

    exclude = [(0, 0, 0), (1, 1, 1)] + [
        tuple(int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)) for h in PAIRING_COLORS
    ]
    extra = distinctipy.get_colors(
        n - len(PAIRING_COLORS), exclude_colors=exclude, rng=42
    )
    return PAIRING_COLORS + [
        "#%02X%02X%02X" % tuple(int(round(c * 255)) for c in rgb) for rgb in extra
    ]


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
    return [
        (run.replace("ah group_switching managed by ", ""), sims[run], color)
        for run, color in zip(runs, pairing_colors(len(runs)))
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


def lineplot(ax, human, sims, stat, xlabel, ylabel, marker=None):
    """Overlaid lines of `stat(frame)` (a pd.Series, index -> value);
    pass a marker for sparse x-axes."""
    for label, df, color, lw in _sources(human, sims):
        s = stat(df)
        ax.plot(
            s.index, s.values, color=color, linewidth=lw, label=label, marker=marker
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _sources(human, sims):
    """Human first (black, heavier), then the pairings in fixed order."""
    out = [("human", human, HUMAN_COLOR, 2.2)]
    out += [(label, df, color, 1.4) for label, df, color in series(sims)]
    return out


# -- Contribution (C) --------------------------------------------------

_C = ContributionMetrics()


@plot("CA_hist")
def ca_hist(ax, human, sims):
    prob_hist(
        ax,
        human,
        sims,
        _C.ca,
        bins=np.linspace(0, 20, 21),
        xlabel="participant mean contribution",
    )


@plot("CB_line")
def cb_line(ax, human, sims):
    lineplot(ax, human, sims, _C.cb, "round", "mean contribution")


@plot("CC_hist")
def cc_hist(ax, human, sims):
    prob_hist(
        ax,
        human,
        sims,
        _C.cc,
        bins=np.linspace(0, 20, 21),
        xlabel="group mean contribution",
    )


@plot("CD_hist")
def cd_hist(ax, human, sims):
    prob_hist(
        ax,
        human,
        sims,
        _C.cd,
        bins=np.arange(-0.5, 21.5),
        xlabel="contribution",
    )


@plot("CE_hist")
def ce_hist(ax, human, sims):
    prob_hist(
        ax,
        human,
        sims,
        _C.ce,
        bins=np.linspace(-20, 20, 41),
        xlabel="signed group contribution difference",
    )


@plot("CE_std_line")
def ce_std_line(ax, human, sims):
    lineplot(
        ax,
        human,
        sims,
        lambda df: _C.ce(df).groupby("round_number").std(),
        "round",
        "std of group difference across games",
    )


@plot("CF_line")
def cf_line(ax, human, sims):
    # share at 0 solid, share at 20 dashed, one color per source
    for label, df, color, lw in _sources(human, sims):
        shares = _C.cf(df).unstack()
        ax.plot(
            shares.index,
            shares["share_at_0"],
            color=color,
            linewidth=lw,
            label=label,
        )
        ax.plot(
            shares.index,
            shares["share_at_20"],
            color=color,
            linewidth=lw,
            linestyle="--",
        )
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color="gray", linestyle="-"),
        Line2D([], [], color="gray", linestyle="--"),
    ]
    labels += ["share at 0", "share at 20"]
    ax.legend(handles, labels, frameon=False, fontsize=8)
    ax.set_xlabel("round")
    ax.set_ylabel("share of contributions")


# -- Switching (S) ------------------------------------------------------

_S = SwitchingMetrics()


@plot("SB_line")
def sb_line(ax, human, sims):
    lineplot(
        ax,
        human,
        sims,
        _S.sb,
        "switching opportunity (decision round)",
        "switch rate",
        marker="o",
    )
    ax.set_xticks([3, 7, 11, 15, 19])


@plot("SC_hist")
def sc_hist(ax, human, sims):
    prob_hist(
        ax,
        human,
        sims,
        _S.sc,
        bins=np.arange(3.5, 9.5),
        xlabel="size of the larger group",
    )


@plot("SC_line")
def sc_line(ax, human, sims):
    lineplot(
        ax,
        human,
        sims,
        lambda df: _S.sc(df).groupby("round_number").mean(),
        "round",
        "size of the larger group (mean over games)",
    )


# -- Punishment (P) ------------------------------------------------------

_P = PunishmentMetrics()


@plot("PA_hist")
def pa_hist(ax, human, sims):
    prob_hist(
        ax,
        human,
        sims,
        _P.pa,
        bins=np.arange(-0.5, 31.5),
        xlabel="punishment",
        log_y=True,
    )


@plot("PB_line")
def pb_line(ax, human, sims):
    lineplot(ax, human, sims, _P.pb, "round", "mean punishment")


@plot("PC_line")
def pc_line(ax, human, sims):
    lineplot(
        ax,
        human,
        sims,
        _P.pc,
        "round",
        "share of punishments at zero",
    )


# -- Responses (R) -------------------------------------------------------

_R = ResponseMetrics()

RCA_ORDER = [
    "no_switch_allowed",
    "switched",
    "chose_to_stay",
    "stayed_comp_changed",
]


def _stratum_stats(obs, order, center="median"):
    """center (median or mean) / q25 / q75 of observations per stratum
    (index level 0)."""
    grouped = obs.groupby(level=0)
    mid = grouped.mean() if center == "mean" else grouped.median()
    return (
        mid.reindex(order),
        grouped.quantile(0.25).reindex(order),
        grouped.quantile(0.75).reindex(order),
    )


def band_lineplot(ax, human, sims, obs, order, xlabel, ylabel, center="median"):
    """Central line with an IQR band per source over categorical strata;
    sources are dodged slightly on x so integer-quantised centers that
    coincide stay visible."""
    x = np.arange(len(order))
    sources = _sources(human, sims)
    for i, (label, df, color, lw) in enumerate(sources):
        dodge = (i - (len(sources) - 1) / 2) * 0.05
        mid, q25, q75 = _stratum_stats(obs(df), order, center)
        ax.plot(
            x + dodge, mid.values, color=color, linewidth=lw, marker="o", label=label
        )
        ax.fill_between(
            x + dodge, q25.values, q75.values, color=color, alpha=0.12, linewidth=0
        )
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{ylabel} ({center}, IQR band)")


@plot("RCA_bar")
def rca_bar(ax, human, sims):
    sources = _sources(human, sims)
    x = np.arange(len(RCA_ORDER))
    width = 0.8 / len(sources)
    for i, (label, df, color, _) in enumerate(sources):
        med, q25, q75 = _stratum_stats(_R.rca(df), RCA_ORDER)
        ax.bar(
            x - 0.4 + width * (i + 0.5),
            med.values,
            width * 0.9,
            color=color,
            label=label,
            yerr=[(med - q25).values, (q75 - med).values],
            error_kw={"ecolor": color, "linewidth": 1.8, "alpha": 0.9},
            capsize=3,
        )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("_", "\n") for o in RCA_ORDER], fontsize=8)
    ax.set_xlabel("round type")
    ax.set_ylabel("contribution change (median, IQR)")


def _rcb_obs(df):
    pop = _R._rcb_population(df)
    return pop.set_index(pop["rate_bin"].astype(str))["dc"]


@plot("RCB_line")
def rcb_line(ax, human, sims):
    # mean center: matches the RCB metric, which compares bin means
    band_lineplot(
        ax,
        human,
        sims,
        _rcb_obs,
        RCB_LABELS,
        "punishment rate (punishment per point of shortfall)",
        "contribution change",
        center="mean",
    )


@plot("RCC_bar")
def rcc_bar(ax, human, sims):
    sources = _sources(human, sims)
    x = np.arange(2)
    width = 0.8 / len(sources)
    for i, (label, df, color, _) in enumerate(sources):
        d = _R._with_dc(df)
        full = d[(d["contribution"] == 20) & d["dc"].notna() & d["punishment"].notna()]
        means = [
            full.loc[full["punishment"] > 0, "dc"].mean(),
            full.loc[full["punishment"] == 0, "dc"].mean(),
        ]
        ax.bar(
            x - 0.4 + width * (i + 0.5), means, width * 0.9, color=color, label=label
        )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["punished", "unpunished"])
    ax.set_xlabel("punishment status of full contributors")
    ax.set_ylabel("mean contribution change")


@plot("RCD_scatter")
def rcd_scatter(ax, human, sims):
    for label, df, color, lw in _sources(human, sims):
        events = _R._switch_events(df).dropna(subset=["dc", "receiving_mean"])
        gap = events["receiving_mean"] - events["contribution"]
        ax.scatter(gap, events["dc"], s=6, alpha=0.15, color=color, edgecolors="none")
        slope = gap.cov(events["dc"]) / gap.var()
        intercept = events["dc"].mean() - slope * gap.mean()
        xs = np.array([gap.min(), gap.max()])
        ax.plot(
            xs,
            intercept + slope * xs,
            color=color,
            linewidth=lw,
            label=f"{label} (pull {slope:.2f})",
        )
    ax.set_xlabel("gap to the receiving group ($\\hat{C} - C_n$)")
    ax.set_ylabel("contribution change ($C_{n+1} - C_n$)")


@plot("RSA_line")
def rsa_line(ax, human, sims):
    x = np.arange(len(RSA_LABELS))
    for label, df, color, lw in _sources(human, sims):
        rates = _R.rsa(df).reindex(RSA_LABELS)
        ax.plot(x, rates.values, color=color, linewidth=lw, marker="o", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(RSA_LABELS)
    ax.set_xlabel("punishment received at the decision round")
    ax.set_ylabel("switch rate")


@plot("RPA_line")
def rpa_line(ax, human, sims):
    band_lineplot(
        ax,
        human,
        sims,
        _R.rpa,
        RPA_LABELS,
        "contribution bin",
        "punishment",
    )


@plot("RPB_line")
def rpb_line(ax, human, sims):
    band_lineplot(
        ax,
        human,
        sims,
        _R.rpb,
        RPB_LABELS,
        "group size",
        "punishment",
    )


def plot_all(human, sims, out_dir):
    """Render every registered figure; returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, fn in PLOTS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fn(ax, human, sims)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.get_legend() is None and ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{name}.jpg")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
