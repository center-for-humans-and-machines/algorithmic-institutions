"""Render figures from an intervention-probe CSV.

Reads ``scenarios.csv`` produced by ``aimanager.simulation.intervention_probe``
and saves PNGs under ``plots/counterfactual/``. Output split by mode:

- Absolute-mode scenarios (``new_value`` set): one bar-chart figure per
  scenario, with one bar group per chosen episode (baseline / treatment /
  real (pilot)).
- Factor-mode scenarios (``factor`` set): one combined bar + trend figure
  per ``(feature, selector, t_star)`` group, aggregating across episodes.
  Bars show the mean baseline / treatment / real at each factor level
  (with across-episode std error bars); a line connects the treatment
  means as the dose-response trend.

Each figure has 2 panels (intervened feature at t*, cross-feature at
t*+1) plus a 3rd switch-probability panel when t*+1 is a decision round.

Usage:
    python scripts/plotting/plot_intervention_scenarios.py \\
        --csv artifacts/counterfactual/{absolute,factor}/scenarios.csv \\
        [--outdir plots/counterfactual]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _bars(ax, x, width, baseline, treatment, real, b_std=None, t_std=None):
    ax.bar(
        x - width,
        baseline,
        width,
        yerr=b_std,
        label="baseline",
        color="C0",
        capsize=3,
    )
    ax.bar(x, treatment, width, yerr=t_std, label="treatment", color="C1", capsize=3)
    ax.bar(x + width, real, width, label="real (pilot)", color="C2", alpha=0.7)


def plot_scenario(group, outpath):
    """``group`` is a per-scenario DataFrame slice (rows = chosen episodes)."""
    g = group.sort_values("ep").reset_index(drop=True)
    feature = g["feature"].iloc[0]
    is_decision = bool(g["is_decision"].iloc[0])
    eps = g["ep"].tolist()
    n = len(eps)

    # Panels: intervened feature at t* (own@t*), same feature at t*+1
    # (own@t*+1 — persistence in its own channel), cross-feature at t*+1
    # (cross-channel response), and switch(t*+1) when t*+1 is a decision
    # round.
    if feature == "punishment":
        own_unit, cross_unit = "punishment", "contribution"
        own_t = ("pun_t", "real_pun_t")
        own_t1 = ("pun_t1", "real_pun_t1")
        cross_t1 = ("contrib_t1", "real_contrib_t1")
    else:
        own_unit, cross_unit = "contribution", "punishment"
        own_t = ("contrib_t", "real_contrib_t")
        own_t1 = ("contrib_t1", "real_contrib_t1")
        cross_t1 = ("pun_t1", "real_pun_t1")

    panels = [
        (*own_t, f"focal {own_unit}(t*)", own_unit),
        (*own_t1, f"focal {own_unit}(t*+1)", own_unit),
        (*cross_t1, f"focal {cross_unit}(t*+1)", cross_unit),
    ]
    if is_decision:
        panels.append(
            (
                "switch_t1",
                "real_switch_t1",
                "focal P(switch)(t*+1)",
                "switch probability",
            )
        )
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.2))
    x = np.arange(n)
    width = 0.28

    for ax, (metric, real_col, title, unit) in zip(axes, panels):
        if metric == "switch_t1":
            real_vals = [1.0 if bool(s) else 0.0 for s in g[real_col]]
            _bars(
                ax,
                x,
                width,
                g[f"{metric}_baseline_mean"].tolist(),
                g[f"{metric}_treatment_mean"].tolist(),
                real_vals,
            )
            ax.set_ylim(0, 1)
        else:
            _bars(
                ax,
                x,
                width,
                g[f"{metric}_baseline_mean"].tolist(),
                g[f"{metric}_treatment_mean"].tolist(),
                g[real_col].tolist(),
                g[f"{metric}_baseline_std"].tolist(),
                g[f"{metric}_treatment_std"].tolist(),
            )
        ax.set_title(title)
        ax.set_ylabel(unit)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(eps)
        ax.set_xlabel("episode")
        ax.legend(loc="upper right", fontsize=8)

    sel = g["selector"].iloc[0]
    sel_str = (
        f"selector={sel} "
        if isinstance(sel, str) and sel and sel.lower() != "nan"
        else ""
    )
    factor = g["factor"].iloc[0] if "factor" in g.columns else None
    new_value = g["new_value"].iloc[0] if "new_value" in g.columns else None
    if pd.notna(factor):
        mod_str = f"factor={factor}"
    else:
        mod_str = f"new_value={new_value}"
    fig.suptitle(
        f"{g['scenario'].iloc[0]}  |  t*={g['t_star'].iloc[0]} "
        f"({'decision' if is_decision else 'NOT decision'})  |  "
        f"feature={feature} {sel_str}{mod_str}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_factor_group(group, outpath):
    """Per-group bar + trend plot for factor-mode dose-response.

    ``group`` is a DataFrame slice covering multiple factor scenarios sharing
    the same (feature, target, selector, t_star). For each factor level:
    bars show mean baseline / treatment / real (averaged across episodes,
    with across-episode std error bars), and a line connects the treatment
    means as the dose-response trend.
    """
    g = group.sort_values(["factor", "ep"]).copy()
    feature = g["feature"].iloc[0]
    is_decision = bool(g["is_decision"].iloc[0])
    t_star = int(g["t_star"].iloc[0])
    factors = sorted(g["factor"].unique().tolist())

    if feature == "punishment":
        own_t = ("pun_t", "real_pun_t", f"focal pun(t*={t_star})", "punishment")
        own_t1 = ("pun_t1", "real_pun_t1", f"focal pun(t*+1={t_star + 1})", "punishment")
        cross_t1 = (
            "contrib_t1",
            "real_contrib_t1",
            f"focal contrib(t*+1={t_star + 1})",
            "contribution",
        )
    else:
        own_t = (
            "contrib_t",
            "real_contrib_t",
            f"focal contrib(t*={t_star})",
            "contribution",
        )
        own_t1 = (
            "contrib_t1",
            "real_contrib_t1",
            f"focal contrib(t*+1={t_star + 1})",
            "contribution",
        )
        cross_t1 = (
            "pun_t1",
            "real_pun_t1",
            f"focal pun(t*+1={t_star + 1})",
            "punishment",
        )

    # Baseline lives at x=1.0 (no perturbation). Insert that position
    # into the factor axis between the < 1 and > 1 factor groups.
    x_factors = sorted(set(factors) | {1.0})
    x = np.arange(len(x_factors))
    treatment_idx = [i for i, f in enumerate(x_factors) if f != 1.0]
    baseline_idx = x_factors.index(1.0)

    fig, axes = plt.subplots(1, 4, figsize=(4.6 * 4, 4.4))

    def _agg_treatment(metric):
        # per-factor across-episode mean + std of treatment_mean
        means, stds = [], []
        for f in factors:
            sub = g[g["factor"] == f]
            col = f"{metric}_treatment_mean"
            means.append(sub[col].mean())
            stds.append(sub[col].std() if len(sub) > 1 else 0.0)
        return means, stds

    def _baseline(metric):
        # Baseline is factor-independent — aggregate across all rows.
        col = f"{metric}_baseline_mean"
        return g[col].mean(), (g[col].std() if g[col].nunique() > 1 else 0.0)

    def _panel(ax, metric, real_col, title, ylabel, ylim=None, show_yerr=True):
        t_mean, t_std = _agg_treatment(metric)
        b_mean, b_std = _baseline(metric)

        # Assemble per-position values: baseline at the x=1.0 slot, treatment
        # at the others. Trend line connects all (including baseline) so
        # the reader sees one continuous response curve.
        full_mean, full_yerr = [], []
        for f in x_factors:
            if f == 1.0:
                full_mean.append(b_mean)
                full_yerr.append(b_std if show_yerr else 0.0)
            else:
                idx = factors.index(f)
                full_mean.append(t_mean[idx])
                full_yerr.append(t_std[idx] if show_yerr else 0.0)

        # Treatment bars (orange).
        ax.bar(
            [x[i] for i in treatment_idx],
            [full_mean[i] for i in treatment_idx],
            0.5,
            yerr=[full_yerr[i] for i in treatment_idx],
            color="C1",
            capsize=3,
            alpha=0.85,
            label="treatment",
        )
        # Baseline bar (blue) at x=1.0.
        ax.bar(
            x[baseline_idx],
            full_mean[baseline_idx],
            0.5,
            yerr=full_yerr[baseline_idx],
            color="C0",
            capsize=3,
            alpha=0.85,
            label="baseline",
        )
        # Trend line through all points.
        ax.plot(
            x,
            full_mean,
            marker="o",
            color="darkorange",
            linewidth=2,
            label="trend",
            zorder=5,
        )

        # Real pilot reference — single horizontal dashed line at the
        # across-episode mean. No band: real is one observation per
        # (episode, focal); spread across episodes is heterogeneity, not
        # measurement noise.
        real_mean = g[real_col].mean()
        ax.axhline(
            real_mean,
            color="C2",
            linestyle="--",
            linewidth=1.5,
            label="real (pilot)",
            alpha=0.85,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(["baseline" if f == 1.0 else f"x{f}" for f in x_factors])
        ax.set_xlabel("factor")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # Panel 1 = intervened metric at t* — deterministic per (episode,
    # factor); no error bars. Panels 2-4 = AH predictions at t*+1 (own-
    # feature persistence, cross-feature response, switch), show
    # across-episode std as error bars.
    _panel(axes[0], *own_t, show_yerr=False)
    _panel(axes[1], *own_t1, show_yerr=True)
    _panel(axes[2], *cross_t1, show_yerr=True)
    sw_title = f"focal P(switch)(t*+1={t_star + 1})"
    if not is_decision:
        sw_title += "  (NOT a decision round)"
    _panel(
        axes[3],
        "switch_t1",
        "real_switch_t1",
        sw_title,
        "switch probability",
        ylim=(0, 1),
        show_yerr=True,
    )

    axes[0].legend(fontsize=8, loc="upper left")

    selector = g["selector"].iloc[0]
    fig.suptitle(
        f"dose-response  |  feature={feature}  selector={selector}  "
        f"t*={t_star} ({'decision' if is_decision else 'NOT decision'})  |  "
        f"{len(factors)} factor levels × {g['ep'].nunique()} episodes",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_group_scenario(group, outpath):
    """One figure per group-target scenario.

    Aggregates across the K chosen episodes into a single bar group per
    panel: baseline / treatment / real (mean ± across-episode std).
    Layout matches plot_scenario: own@t*, own@t*+1, cross@t*+1, plus
    switch@t*+1 when t*+1 is a decision round.
    """
    g = group
    feature = g["feature"].iloc[0]
    is_decision = bool(g["is_decision"].iloc[0])
    t_star = int(g["t_star"].iloc[0])

    if feature == "punishment":
        own_unit, cross_unit = "punishment", "contribution"
        own_t = ("pun_t", "real_pun_t")
        own_t1 = ("pun_t1", "real_pun_t1")
        cross_t1 = ("contrib_t1", "real_contrib_t1")
    else:
        own_unit, cross_unit = "contribution", "punishment"
        own_t = ("contrib_t", "real_contrib_t")
        own_t1 = ("contrib_t1", "real_contrib_t1")
        cross_t1 = ("pun_t1", "real_pun_t1")

    panels = [
        (*own_t, f"{own_unit}(t*={t_star})", own_unit, None),
        (*own_t1, f"{own_unit}(t*+1={t_star + 1})", own_unit, None),
        (*cross_t1, f"{cross_unit}(t*+1={t_star + 1})", cross_unit, None),
    ]
    if is_decision:
        panels.append(
            (
                "switch_t1",
                "real_switch_t1",
                f"P(switch)(t*+1={t_star + 1})",
                "switch probability",
                (0, 1),
            )
        )
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 4.2))
    x = np.array([0, 1, 2])
    width = 0.6

    for ax, (metric, real_col, title, unit, ylim) in zip(axes, panels):
        b_mean = g[f"{metric}_baseline_mean"].mean()
        b_std = g[f"{metric}_baseline_mean"].std() if len(g) > 1 else 0.0
        t_mean = g[f"{metric}_treatment_mean"].mean()
        t_std = g[f"{metric}_treatment_mean"].std() if len(g) > 1 else 0.0
        r_mean = g[real_col].mean()
        r_std = g[real_col].std() if len(g) > 1 else 0.0

        ax.bar(0, b_mean, width, yerr=b_std, color="C0", capsize=4, label="baseline")
        ax.bar(
            1, t_mean, width, yerr=t_std, color="C1", capsize=4, label="treatment"
        )
        ax.bar(
            2,
            r_mean,
            width,
            yerr=r_std,
            color="C2",
            alpha=0.7,
            capsize=4,
            label="real (pilot)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["baseline", "treatment", "real"])
        ax.set_title(title)
        ax.set_ylabel(unit)
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0].legend(fontsize=8, loc="upper right")

    sel = g["selector"].iloc[0] if "selector" in g.columns else None
    factor = g["factor"].iloc[0] if "factor" in g.columns else None
    new_value = g["new_value"].iloc[0] if "new_value" in g.columns else None
    mod_str = f"factor={factor}" if pd.notna(factor) else f"new_value={new_value}"
    fig.suptitle(
        f"{g['scenario'].iloc[0]}  |  group target  |  "
        f"feature={feature}  selector={sel}  {mod_str}  |  "
        f"t*={t_star} ({'decision' if is_decision else 'NOT decision'})  |  "
        f"averaged over {len(g)} episodes × n_seeds",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default="plots/counterfactual")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} rows from {args.csv}")

    has_target = "target" in df.columns
    group_df = df[df["target"] == "group"] if has_target else df.iloc[0:0]
    rest = df[df["target"] != "group"] if has_target else df

    absolute_df = rest[rest["factor"].isna()] if "factor" in rest.columns else rest
    factor_df = (
        rest[rest["factor"].notna()] if "factor" in rest.columns else rest.iloc[0:0]
    )

    # Per-scenario bar plots for absolute-mode individual scenarios.
    for name, g in absolute_df.groupby("scenario"):
        outpath = os.path.join(args.outdir, f"{name}.png")
        plot_scenario(g, outpath)
        print(f"saved {outpath}")

    # Per-(feature, selector, t_star) bar+trend plots for factor-mode
    # individual scenarios — one figure aggregates the dose-response sweep.
    for (feature, selector, t_star), g in factor_df.groupby(
        ["feature", "selector", "t_star"]
    ):
        name = f"dose_response_{feature}_{selector}_t{t_star}.png"
        outpath = os.path.join(args.outdir, name)
        plot_factor_group(g, outpath)
        print(f"saved {outpath}")

    # Per-scenario aggregated plots for group-target scenarios.
    for name, g in group_df.groupby("scenario"):
        outpath = os.path.join(args.outdir, f"{name}.png")
        plot_group_scenario(g, outpath)
        print(f"saved {outpath}")


if __name__ == "__main__":
    main()
