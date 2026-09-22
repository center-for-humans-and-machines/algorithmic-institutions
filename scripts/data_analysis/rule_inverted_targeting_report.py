"""Does punishing the WRONG people cost anything in this world?

The learned RL managers came out inverted -- they punish full contributors
hardest and spare free-riders. An intervention probe found the contributor
model's targeting signal real but weak, and close to inert at the ceiling
where the learned seeds aim. That is a claim about consequences, and this
script scores the simulation that tests it: an inverted rule run in the
COMPETING setting beside the correctly-targeted rule and never-punishing.

Everything about the accounting, the bootstrap and the seat controls is
inherited from `rule_vs_clone_paired_report`, deliberately, so the rows drop
straight into that arm's tables. What is added here is:

* `mirror_match` -- the realised firing rate and punishment spend of each
  rule, so a reader can see whether the mirror is a fair one rather than
  taking the name's word for it;
* `focal_contrast` -- one focal seat against another focal seat with the
  SAME rival in the SAME seat, which is what separates the cost of punishing
  from the cost of punishing in the wrong direction.

Usage:
    python scripts/data_analysis/rule_inverted_targeting_report.py \
        plots/simulation/26_rule_inverted_targeting_s42 --tag s42
    python scripts/data_analysis/rule_inverted_targeting_report.py \
        plots/simulation/26_rule_inverted_targeting_s{42,43,44} --tag pooled
    python scripts/data_analysis/rule_inverted_targeting_report.py \
        --aggregate s42 s43 s44
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rule_vs_clone_paired_report as paired  # noqa: E402

OUT_DIR = "plots/data_analysis/evaluation/rule_inverted_targeting"

CLONE = paired.CLONE
NEVER = paired.NEVER
RIVALS = paired.RIVALS
SERIES = paired.SERIES
MUTED = paired.MUTED
INK = paired.INK
GRID = paired.GRID

INV_LEVEL = "inv_thr11_p10"  # exact reflection of thr9_p10 under c -> 20 - c
INV_SPEND = "inv_thr7_p10"  # matched instead on realised punishment spend
CORRECT = "thr9_p10"
FOCALS = [INV_LEVEL, INV_SPEND, CORRECT, NEVER]

# colour follows the ENTITY, fixed order, never cycled
COLOR = {
    INV_LEVEL: SERIES[1],
    INV_SPEND: SERIES[3],
    CORRECT: SERIES[0],
    NEVER: SERIES[2],
    CLONE: MUTED,
}
LABEL = {
    INV_LEVEL: "inv_thr11_p10 (inverted, level-matched)",
    INV_SPEND: "inv_thr7_p10 (inverted, spend-matched)",
    CORRECT: "thr9_p10 (correctly targeted)",
    NEVER: "never (no punishment)",
    CLONE: "ah_punisher (the clone)",
}

# which contribution levels each rule fires on, from its definition
FIRES_ON = {
    CORRECT: lambda c: c <= 9,
    INV_LEVEL: lambda c: c >= 11,
    INV_SPEND: lambda c: c >= 7,
    NEVER: lambda c: np.zeros_like(c, dtype=bool),
}


def check_dispatch(df):
    """Each seat must carry its manager's signature in the output.

    Extends the parent's check with the two inverted rules: an inverted seat
    punishes only 0 or its amount, punishes NOTHING at or below its
    threshold minus one, and punishes EVERY valid cell at or above it. A
    seat that had the forward rule, or static dispatch, fails all three."""

    def inv_sig(threshold, amount):
        def f(d):
            c = d["contribution"].to_numpy()
            p = d["punishment"].to_numpy()
            return bool(
                set(np.unique(p)) <= {0, amount}
                and (p[c < threshold] == 0).all()
                and (p[c >= threshold] == amount).all()
            )

        return f

    sig = {
        NEVER: lambda d: set(d["punishment"].unique()) <= {0},
        CORRECT: lambda d: bool(
            set(d["punishment"].unique()) <= {0, 10}
            and (
                d["punishment"].to_numpy()[d["contribution"].to_numpy() <= 9] == 10
            ).all()
        ),
        INV_LEVEL: inv_sig(11, 10),
        INV_SPEND: inv_sig(7, 10),
    }
    out = []
    for (pairing, gid), sub in df.groupby(["pairing", "group_id"]):
        seat = "focal" if gid == 0 else "rival"
        m = paired.seat_manager(pairing, seat)
        if m not in sig or not len(sub):
            continue
        holds = sig[m](sub)
        out.append({"pairing": pairing, "seat": seat, "manager": m, "holds": holds})
        assert holds, f"{m} signature violated on the {seat} seat of {pairing}"
    n = len(out)
    print(f"dispatch check: {n}/{n} seats carry their manager's signature")
    return pd.DataFrame(out)


def mirror_match(df, out_dir, tag):
    """Is the inverted rule a FAIR mirror of the correctly-targeted one?

    A mirror that fires on a different share of cells, or spends a different
    amount of punishment, confounds direction with intensity. Two senses of
    "mirror" are reported because they disagree in this world:

    * levels -- how many of the 21 contribution levels the rule fires on,
      a property of the DEFINITION;
    * spend -- the realised firing rate and mean punishment per member-round,
      a property of the RUN, and endogenous, since the rule moves the
      contributions it then reads.

    `fires_counterfactual` is the third column that makes the two
    comparable: the rate at which each rule WOULD fire on the untreated
    contribution distribution (the `never_vs_never` control of this same
    run), which is how the spend-matched threshold was chosen ex ante."""
    base = df[df["pairing"] == f"{NEVER}_vs_{NEVER}"]["contribution"].to_numpy()
    rows = []
    for (pairing, gid), sub in df.groupby(["pairing", "group_id"]):
        seat = "focal" if gid == 0 else "rival"
        m = paired.seat_manager(pairing, seat)
        if m not in FIRES_ON or not len(sub):
            continue
        valid = sub["contribution_valid"].astype(bool).to_numpy()
        p = sub["punishment"].to_numpy().astype(float)
        rows.append(
            {
                "pairing": pairing,
                "seat": seat,
                "manager": m,
                "n_levels_fired": int(sum(FIRES_ON[m](np.arange(21)))),
                "fires_counterfactual": float(FIRES_ON[m](base).mean()),
                "fires_realised": float((p > 0).mean()),
                "fires_realised_valid": float((p[valid] > 0).mean()),
                "mean_punishment": float(p.mean()),
                "mean_punishment_valid": float(p[valid].mean()),
            }
        )
    out = pd.DataFrame(rows)
    p = os.path.join(out_dir, f"mirror_match_{tag}.csv")
    out.to_csv(p, index=False)
    print(f"\nwrote {p}")
    print("\n=== is the mirror fair? realised spend of each rule, per seat ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return out


def focal_contrast(pe, pairs, label):
    """Focal seat of A minus focal seat of B, same rival, same seat.

    This is the contrast that isolates DIRECTION. Both pairings put a
    manager in group 0 against the same rival in group 1, so the rival, the
    seat and the protocol are held fixed and only the focal rule differs.
    The two come from different worlds, so the interval is unpaired -- the
    same construction the parent used for `vs_control`."""
    idx = pe.set_index(["pairing", "seat", "episode"])
    rows = []
    for a, b in pairs:
        fa = idx.xs((a, "focal"), level=("pairing", "seat")).sort_index()
        fb = idx.xs((b, "focal"), level=("pairing", "seat")).sort_index()
        row = {"contrast": label, "a": a, "b": b}
        for m in paired.METRICS:
            d, lo, hi = paired.unpaired_ci(fa[m], fb[m])
            row[f"d_{m}"], row[f"d_{m}_lo"], row[f"d_{m}_hi"] = d, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def decision(pe, out_dir, tag):
    """The two contrasts the arm was run to make, and the levels behind them.

    1. inverted MINUS never-punishing -- is the inversion penalised at all?
    2. inverted MINUS correctly-targeted -- does the DIRECTION of punishment
       matter, holding the amount of punishing roughly fixed?

    Contrast 1 alone cannot answer the question the arm poses, because
    punishing at all costs members in this world. Contrast 2 is what
    separates the cost of punishing from the cost of punishing the wrong
    people, and it is the one to read for direction."""
    frames = []
    for rival in RIVALS:
        vs_never = [
            (f"{f}_vs_{rival}", f"{NEVER}_vs_{rival}") for f in (INV_LEVEL, INV_SPEND)
        ]
        vs_never.append((f"{CORRECT}_vs_{rival}", f"{NEVER}_vs_{rival}"))
        d1 = focal_contrast(pe, vs_never, "minus_never")
        vs_correct = [
            (f"{f}_vs_{rival}", f"{CORRECT}_vs_{rival}") for f in (INV_LEVEL, INV_SPEND)
        ]
        d2 = focal_contrast(pe, vs_correct, "minus_correct")
        for d in (d1, d2):
            d["rival"] = rival
            frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    p = os.path.join(out_dir, f"decision_{tag}.csv")
    out.to_csv(p, index=False)
    print(f"\nwrote {p}")
    for label, title in [
        ("minus_never", "focal seat MINUS the never-punish seat (same rival)"),
        ("minus_correct", "focal seat MINUS the thr9_p10 seat (same rival)"),
    ]:
        print(f"\n=== {title} ===")
        for _, r in out[out["contrast"] == label].iterrows():
            print(
                f"  rival {r['rival']:<12s} {r['a'].split('_vs_')[0]:>14s}  "
                f"size {r['d_group_size']:+6.2f} "
                f"[{r['d_group_size_lo']:+6.2f}, {r['d_group_size_hi']:+6.2f}]   "
                f"pool {r['d_pool_corr']:+7.2f} "
                f"[{r['d_pool_corr_lo']:+7.2f}, {r['d_pool_corr_hi']:+7.2f}]   "
                f"per-member {r['d_share_corr']:+6.2f} "
                f"[{r['d_share_corr_lo']:+6.2f}, {r['d_share_corr_hi']:+6.2f}]"
            )
    return out


def levels(summary, out_dir, tag):
    """The headline table: what each focal seat actually holds and produces."""
    cols = [
        "pairing",
        "manager",
        "rival",
        "group_size",
        "group_size_lo",
        "group_size_hi",
        "pool_corr",
        "pool_corr_lo",
        "pool_corr_hi",
        "share_corr",
        "share_corr_lo",
        "share_corr_hi",
        "mean_c",
        "mean_p",
        "payoff_corr",
    ]
    out = summary[summary["seat"] == "focal"][cols].copy()
    out = out[out["manager"].isin(FOCALS)].sort_values(["rival", "manager"])
    p = os.path.join(out_dir, f"levels_{tag}.csv")
    out.to_csv(p, index=False)
    print(f"\nwrote {p}")
    print("\n=== the focal seat: members, pool, pool per member ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return out


# ---------------------------------------------------------------------- #
# figures
# ---------------------------------------------------------------------- #


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)


def fig_decision(lv, out_dir, tag):
    """Dot-and-interval, one panel per rival, one row per focal rule.

    A dot plot rather than bars: the quantity carrying the argument is an
    interval, and bars invite reading the length rather than the position."""
    metrics = [
        ("group_size", "group size (members held, of 8)"),
        ("pool_corr", "common good, group total per round"),
        ("share_corr", "common good per member per round"),
    ]
    fig, axes = plt.subplots(
        len(RIVALS), 3, figsize=(13.5, 6.2), sharey=True, constrained_layout=True
    )
    order = [CORRECT, INV_SPEND, INV_LEVEL, NEVER]
    ypos = {m: i for i, m in enumerate(order[::-1])}
    for r_i, rival in enumerate(RIVALS):
        sub = lv[lv["rival"] == rival].set_index("manager")
        for m_i, (metric, title) in enumerate(metrics):
            ax = axes[r_i][m_i]
            for m in order:
                if m not in sub.index:
                    continue
                y = ypos[m]
                v = sub.loc[m, metric]
                lo, hi = sub.loc[m, f"{metric}_lo"], sub.loc[m, f"{metric}_hi"]
                ax.plot(
                    [lo, hi],
                    [y, y],
                    color=COLOR[m],
                    linewidth=2,
                    solid_capstyle="round",
                )
                ax.plot(
                    [v],
                    [y],
                    "o",
                    color=COLOR[m],
                    markersize=9,
                    markeredgecolor="white",
                    markeredgewidth=2,
                    zorder=3,
                )
                ax.annotate(
                    f"{v:.2f}",
                    (v, y),
                    textcoords="offset points",
                    xytext=(0, 11),
                    ha="center",
                    fontsize=8.5,
                    color=INK,
                )
            # the reference the question is posed against
            if NEVER in sub.index:
                ax.axvline(
                    sub.loc[NEVER, metric],
                    color=MUTED,
                    linewidth=1,
                    linestyle=(0, (4, 3)),
                    zorder=0,
                )
            _style(ax)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([m for m in order[::-1]], fontsize=9)
            ax.set_ylim(-0.6, len(order) - 0.4)
            if r_i == 0:
                ax.set_title(title, fontsize=10, color=INK)
            if m_i == 0:
                ax.set_ylabel(f"rival: {rival}", fontsize=10, color=INK, labelpad=8)
    fig.suptitle(
        "Punishing the wrong people, in the competing setting\n"
        "focal seat only; dashed line = never-punish in the same seat; "
        "95% episode bootstrap, 3 seeds",
        fontsize=11.5,
        color=INK,
    )
    p = os.path.join(out_dir, f"inverted_decision_{tag}.jpg")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {p}")


def fig_spend_vs_pool(lv, out_dir, tag):
    """Realised punishment against what the seat keeps.

    If direction were free, the inverted and the correctly-targeted rule at
    the same spend would sit at the same height."""
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True
    )
    for ax, rival in zip(axes, RIVALS):
        sub = lv[lv["rival"] == rival]
        for _, r in sub.iterrows():
            m = r["manager"]
            ax.plot(
                [r["mean_p"]],
                [r["pool_corr"]],
                "o",
                color=COLOR[m],
                markersize=11,
                markeredgecolor="white",
                markeredgewidth=2,
                zorder=3,
            )
            ax.errorbar(
                r["mean_p"],
                r["pool_corr"],
                yerr=[
                    [r["pool_corr"] - r["pool_corr_lo"]],
                    [r["pool_corr_hi"] - r["pool_corr"]],
                ],
                color=COLOR[m],
                linewidth=1.6,
                capsize=0,
                zorder=2,
            )
            ax.annotate(
                m,
                (r["mean_p"], r["pool_corr"]),
                textcoords="offset points",
                xytext=(0, 14),
                ha="center",
                fontsize=8.5,
                color=INK,
            )
        _style(ax)
        ax.set_title(f"rival: {rival}", fontsize=10, color=INK)
        ax.set_xlabel(
            "realised punishment per member per round", fontsize=9.5, color=INK
        )
    axes[0].set_ylabel("common good, group total per round", fontsize=9.5, color=INK)
    fig.suptitle(
        "Same spend, opposite direction: what the seat keeps\n"
        "focal seat, corrected accounting, 95% episode bootstrap",
        fontsize=11.5,
        color=INK,
    )
    p = os.path.join(out_dir, f"inverted_spend_vs_pool_{tag}.jpg")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {p}")


def fig_group_size(series, out_dir, tag):
    """Members held, round by round -- the quantity self-play could not see."""
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True
    )
    for ax, rival in zip(axes, RIVALS):
        for m in [CORRECT, INV_SPEND, INV_LEVEL, NEVER]:
            pairing = f"{m}_vs_{rival}"
            sub = series[(series["pairing"] == pairing) & (series["seat"] == "focal")]
            if not len(sub):
                continue
            sub = sub.sort_values("round_number")
            ax.plot(
                sub["round_number"],
                sub["group_size"],
                color=COLOR[m],
                linewidth=2,
                label=LABEL[m],
                solid_capstyle="round",
            )
            last = sub.iloc[-1]
            ax.annotate(
                m,
                (last["round_number"], last["group_size"]),
                textcoords="offset points",
                xytext=(5, 0),
                ha="left",
                fontsize=8,
                color=INK,
                va="center",
            )
        ax.axhline(4, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
        _style(ax)
        ax.set_title(f"rival: {rival}", fontsize=10, color=INK)
        ax.set_xlabel("round", fontsize=9.5, color=INK)
        ax.set_xlim(0, 27)
    axes[0].set_ylabel("members held (of 8)", fontsize=9.5, color=INK)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle(
        "Every punisher bleeds members; direction barely changes the rate\n"
        "focal seat, dashed line = the 4-of-8 start, 3 seeds pooled",
        fontsize=11.5,
        color=INK,
    )
    p = os.path.join(out_dir, f"inverted_group_size_{tag}.jpg")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dirs", nargs="*")
    ap.add_argument("--tag", default="pooled")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--aggregate", nargs="+")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"grid.linewidth": 0.5, "axes.edgecolor": GRID})

    if args.aggregate:
        paired.aggregate(args.aggregate, args.out_dir)
        return

    df = paired.load(args.sim_dirs)
    check_dispatch(df)

    g = paired.group_rounds(df)
    pe = paired.per_episode(g)
    pe_late = paired.per_episode(g, late=True)

    summary = paired.summarise(pe, pe_late, paired.timeout_stats(df))
    summary.to_csv(os.path.join(args.out_dir, f"summary_{args.tag}.csv"), index=False)

    mirror_match(df, args.out_dir, args.tag)
    lv = levels(summary, args.out_dir, args.tag)
    decision(pe, args.out_dir, args.tag)

    ww = paired.within_world(pe)
    ww.to_csv(os.path.join(args.out_dir, f"within_world_{args.tag}.csv"), index=False)
    print("\n=== focal minus rival, same world, paired by episode ===")
    for _, r in ww.iterrows():
        print(
            f"  {r['focal']:>15s} vs {r['rival']:<12s} "
            f"size {r['d_group_size']:+6.2f} "
            f"[{r['d_group_size_lo']:+6.2f}, {r['d_group_size_hi']:+6.2f}]   "
            f"pool {r['d_pool_corr']:+7.2f} "
            f"[{r['d_pool_corr_lo']:+7.2f}, {r['d_pool_corr_hi']:+7.2f}]"
        )

    wc = paired.vs_control(pe, "focal")
    wc.to_csv(os.path.join(args.out_dir, f"vs_control_{args.tag}.csv"), index=False)
    wr = paired.vs_control(pe, "rival")
    wr.to_csv(
        os.path.join(args.out_dir, f"vs_control_rival_{args.tag}.csv"), index=False
    )
    print("\n=== what the RIVAL seat gains from facing this rule ===")
    for _, r in wr.iterrows():
        print(
            f"  {r['rival']:>12s} facing {r['focal']:<15s} "
            f"size {r['d_group_size']:+6.2f} "
            f"[{r['d_group_size_lo']:+6.2f}, {r['d_group_size_hi']:+6.2f}]   "
            f"pool {r['d_pool_corr']:+7.2f} "
            f"[{r['d_pool_corr_lo']:+7.2f}, {r['d_pool_corr_hi']:+7.2f}]"
        )

    wl = paired.who_leaves(df)
    wl.to_csv(os.path.join(args.out_dir, f"who_leaves_{args.tag}.csv"), index=False)
    print("\n=== who leaves a seat, at the rounds a move is decided ===")
    print(
        wl.sort_values(["seat", "leave_rate"], ascending=[True, False]).to_string(
            index=False, float_format=lambda v: f"{v:.2f}"
        )
    )

    series = paired.round_series(g)
    series.to_csv(
        os.path.join(args.out_dir, f"round_series_{args.tag}.csv"), index=False
    )

    print("\n=== full per-seat summary ===")
    cols = [
        "pairing",
        "seat",
        "manager",
        "group_size",
        "group_size_late",
        "pool_corr",
        "pool_env",
        "share_corr",
        "mean_c",
        "mean_p",
        "payoff_corr",
        "timeout_share_punished",
        "timeout_punishment_share",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    fig_decision(lv, args.out_dir, args.tag)
    fig_spend_vs_pool(lv, args.out_dir, args.tag)
    fig_group_size(series, args.out_dir, args.tag)


if __name__ == "__main__":
    main()
