"""Score a rule and its rival when they share one world, one group each.

The sweep (`rule_manager_sweep_report.py`) ran every manager in self-play:
both seats of a pairing carried the same manager, so a manager's number was
the whole population's and "how big is the group this manager holds" had no
meaning. Here `group_0` carries a focal manager and `group_1` a rival, and
members move between them every `switch_every` rounds -- so the quantities
that matter are per SEAT, and group size is one of them.

Everything is recomputed from contributions and punishments. The trap this
avoids: the env's `common_good` state field is the per-capita SHARE, while
the column of that name in the human data is the undivided pool, and
`per_round.parquet` is written from the env
(notes/autoresearch_log/manager-common-pool-reward.md). Neither column is
read here.

Two accountings, as the sweep used:

* **env** -- punishment aimed at a timed-out player is dropped (it is free),
  and that player's own payoff is discarded.
* **corrected** -- that punishment is charged to the pool, because the
  artificial humans were shown it and reacted to it, and the timed-out
  player is paid `20 - 0 - 0 + share`, as the real game paid them.

Usage:
    python scripts/data_analysis/rule_vs_clone_paired_report.py \\
        plots/simulation/25_rule_vs_clone_paired_s42 --tag s42
    python scripts/data_analysis/rule_vs_clone_paired_report.py \\
        --aggregate s42 s43 s44
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = "plots/data_analysis/evaluation/rule_vs_clone_paired"
ENDOWMENT = 20.0
MPCR = 1.6
LATE_FROM = 16  # "settled" window: rounds 16..23, four switch opportunities in

RUN_RE = re.compile(r"^ah .* managed by (?P<pairing>.+)$")

# validated categorical slots (dataviz reference palette, light mode), the
# same convention the sweep's report uses so the two read as one system.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MUTED = "#b8b7b1"
INK = "#0b0b0b"
GRID = "#d8d7d2"

CLONE = "ah_punisher"
NEVER = "never"
RIVALS = [CLONE, NEVER]

# The sweep's head-to-head arm, common good on the env accounting, both
# groups summed, 300 episodes (log section 3.1). Carried here so the two
# settings can be put side by side without re-running the sweep.
SWEEP_H2H_CG = {
    "prop10": 136.04,
    "thr9_p10": 123.79,
    "thr9_p5": 121.59,
    "human_severity": 116.22,
    "ah_punisher": 111.06,
    "never": 99.63,
}


def split_pairing(name):
    """`{focal}_vs_{rival}` -> (focal, rival). Rivals are a closed set, so
    split on the LAST `_vs_` that leaves a known rival behind."""
    for r in RIVALS:
        if name.endswith(f"_vs_{r}"):
            return name[: -len(f"_vs_{r}")], r
    raise ValueError(f"cannot parse pairing {name!r}")


def group_rounds(df):
    """Per (pairing, episode, round, group) accounting, on both conventions.

    Every (pairing, episode, round, group) cell is emitted even when the
    group is EMPTY -- all eight players can merge into one seat, and a seat
    that has lost every member is a result, not a missing row."""
    valid = df["contribution_valid"].astype(bool)
    d = df.assign(
        c_eff=df["contribution"].where(valid, 0.0).astype(float),
        p_valid=df["punishment"].where(valid, 0.0).astype(float),
        p_all=df["punishment"].astype(float),
        n_valid=valid.astype(float),
    )
    keys = ["pairing", "episode", "round_number", "group_id"]
    g = d.groupby(keys, as_index=False).agg(
        n=("c_eff", "size"),
        n_valid=("n_valid", "sum"),
        sum_c=("c_eff", "sum"),
        sum_p_env=("p_valid", "sum"),
        sum_p_all=("p_all", "sum"),
    )

    # reindex onto the full grid so empty seats appear as size 0
    idx = pd.MultiIndex.from_product(
        [
            sorted(g["pairing"].unique()),
            sorted(g["episode"].unique()),
            sorted(g["round_number"].unique()),
            [0, 1],
        ],
        names=keys,
    )
    g = g.set_index(keys).reindex(idx).fillna(0.0).reset_index()

    nv = g["n_valid"].clip(lower=1)
    g["pool_env"] = MPCR * g["sum_c"] - g["sum_p_env"]
    g["pool_corr"] = MPCR * g["sum_c"] - g["sum_p_all"]
    g["share_corr"] = g["pool_corr"] / nv
    g["payoff_env"] = ENDOWMENT * g["n_valid"] + 0.6 * g["sum_c"] - 2 * g["sum_p_env"]
    g["payoff_corr"] = (
        ENDOWMENT * g["n"] - g["sum_c"] - g["sum_p_all"] + g["n"] * g["pool_corr"] / nv
    )
    empty = g["n_valid"] == 0
    for c in ["pool_env", "pool_corr", "share_corr", "payoff_env", "payoff_corr"]:
        g.loc[empty, c] = 0.0
    g["group_size"] = g["n"]
    g["mean_c"] = np.where(g["n_valid"] > 0, g["sum_c"] / nv, np.nan)
    g["mean_p"] = np.where(g["n_valid"] > 0, g["sum_p_all"] / nv, np.nan)
    g["payoff_corr_pc"] = np.where(g["n"] > 0, g["payoff_corr"] / g["n"].clip(1), 0.0)
    g["seat"] = np.where(g["group_id"] == 0, "focal", "rival")
    return g


METRICS = [
    "group_size",
    "pool_env",
    "pool_corr",
    "share_corr",
    "payoff_env",
    "payoff_corr",
    "payoff_corr_pc",
    "mean_c",
    "mean_p",
]


def per_episode(g, late=False):
    """One value per (pairing, seat, episode): the mean over rounds.

    An episode is the resampling unit throughout -- rounds inside an episode
    are one trajectory and are not independent."""
    d = g[g["round_number"] >= LATE_FROM] if late else g
    return d.groupby(["pairing", "seat", "episode"], as_index=False)[METRICS].mean()


def boot_ci(vals, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    d = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def paired_ci(a, b, n_boot=4000, seed=1):
    """95% interval of mean(a - b) resampling EPISODES, keeping the pair
    together. Focal and rival live in the same episode of the same world, so
    the difference is paired and the paired interval is the honest one."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    draws = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return (
        float(d.mean()),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def unpaired_ci(a, b, n_boot=4000, seed=1):
    """Same construction the sweep used, for across-pairing contrasts."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    da = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    d = da - db
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def seat_manager(pairing, seat):
    focal, rival = split_pairing(pairing)
    return focal if seat == "focal" else rival


def check_dispatch(df):
    """Prove the pairing really put a different manager in each seat.

    Each rule has a signature the data must carry on the seat that holds it
    and must NOT carry on the seat that does not: `never` punishes 0 always,
    the thresholds punish only 0 or their amount, `prop10` punishes exactly
    `20 - c` on cells where the player gave an input. If dispatch were
    static (one manager on all agents) or keyed on the wrong seat, these
    fail."""
    sig = {
        NEVER: lambda d: set(d["punishment"].unique()) <= {0},
        "thr9_p10": lambda d: set(d["punishment"].unique()) <= {0, 10},
        "thr9_p5": lambda d: set(d["punishment"].unique()) <= {0, 5},
        "prop10": lambda d: bool(
            (
                d.loc[d["contribution_valid"].astype(bool), "punishment"]
                == 20 - d.loc[d["contribution_valid"].astype(bool), "contribution"]
            ).all()
        ),
    }
    out = []
    for (pairing, gid), sub in df.groupby(["pairing", "group_id"]):
        seat = "focal" if gid == 0 else "rival"
        m = seat_manager(pairing, seat)
        if m not in sig or not len(sub):
            continue
        holds = sig[m](sub)
        out.append({"pairing": pairing, "seat": seat, "manager": m, "holds": holds})
        assert holds, f"{m} signature violated on the {seat} seat of {pairing}"
    n = len(out)
    print(f"dispatch check: {n}/{n} seats carry their manager's signature")
    return pd.DataFrame(out)


def timeout_stats(df):
    """How much of each seat's punishment lands on players who gave no input.

    Carried forward from the sweep (section 3.6): a contribution-keyed rule
    punishes every timed-out player at full severity because the env serves
    them contribution 0. `auto/free-punishment-fix` closes this at the env;
    it had not landed when this ran, so the rate is reported instead."""
    rows = []
    for (pairing, gid), sub in df.groupby(["pairing", "group_id"]):
        valid = sub["contribution_valid"].astype(bool)
        to = sub[~valid]
        seat = "focal" if gid == 0 else "rival"
        rows.append(
            {
                "pairing": pairing,
                "seat": seat,
                "manager": seat_manager(pairing, seat),
                "timeout_rate": float((~valid).mean()),
                "timeout_share_punished": float(
                    (to["punishment"] > 0).mean() if len(to) else 0.0
                ),
                "timeout_punishment_share": float(
                    to["punishment"].sum() / max(sub["punishment"].sum(), 1e-9)
                ),
            }
        )
    return pd.DataFrame(rows)


def summarise(pe, pe_late, to_df):
    rows = []
    for (pairing, seat), sub in pe.groupby(["pairing", "seat"]):
        late = pe_late[(pe_late["pairing"] == pairing) & (pe_late["seat"] == seat)]
        row = {
            "pairing": pairing,
            "seat": seat,
            "manager": seat_manager(pairing, seat),
            "rival": split_pairing(pairing)[1],
            "n_episodes": int(sub["episode"].nunique()),
        }
        for m in METRICS:
            row[m] = float(sub[m].mean())
            lo, hi = boot_ci(sub[m])
            row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        row["group_size_late"] = float(late["group_size"].mean())
        row["pool_corr_late"] = float(late["pool_corr"].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.merge(to_df, on=["pairing", "seat", "manager"], how="left")


def within_world(pe):
    """focal - rival, inside one world, paired by episode."""
    rows = []
    for pairing, sub in pe.groupby("pairing"):
        f = sub[sub["seat"] == "focal"].set_index("episode").sort_index()
        r = sub[sub["seat"] == "rival"].set_index("episode").sort_index()
        focal, rival = split_pairing(pairing)
        row = {"pairing": pairing, "focal": focal, "rival": rival}
        for m in METRICS:
            d, lo, hi = paired_ci(f[m], r[m])
            row[f"d_{m}"], row[f"d_{m}_lo"], row[f"d_{m}_hi"] = d, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def vs_control(pe):
    """Focal seat against the SAME seat of the symmetric control.

    `prop10_vs_ah_punisher`'s group 0 against `ah_punisher_vs_ah_punisher`'s
    group 0: same seat, same rival, only the rule differs. This is the
    paired setting's analogue of the sweep's margin over the clone."""
    controls = {CLONE: f"{CLONE}_vs_{CLONE}", NEVER: f"{NEVER}_vs_{NEVER}"}
    idx = pe.set_index(["pairing", "seat", "episode"])
    rows = []
    for pairing, sub in pe.groupby("pairing"):
        focal, rival = split_pairing(pairing)
        ctrl = controls[rival]
        if pairing == ctrl:
            continue
        f = sub[sub["seat"] == "focal"].set_index("episode").sort_index()
        c = idx.xs((ctrl, "focal"), level=("pairing", "seat")).sort_index()
        row = {"pairing": pairing, "focal": focal, "rival": rival, "control": ctrl}
        for m in METRICS:
            d, lo, hi = unpaired_ci(f[m], c[m])
            row[f"d_{m}"], row[f"d_{m}_lo"], row[f"d_{m}_hi"] = d, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def round_series(g):
    """Mean of every metric per (pairing, seat, round).

    Group size is a step series -- switching happens only every
    `switch_every` rounds -- so it is read off this, not smoothed."""
    return g.groupby(["pairing", "seat", "round_number"], as_index=False)[
        METRICS
    ].mean()


# ---------------------------------------------------------------------- #
# figures
# ---------------------------------------------------------------------- #


def fig_group_size(series, out_dir, tag):
    """The point of the exercise: does a rule hold members against a rival,
    and does that depend on which rival?"""
    focal = series[series["seat"] == "focal"].copy()
    focal["focal"] = focal["pairing"].map(lambda p: split_pairing(p)[0])
    focal["rival"] = focal["pairing"].map(lambda p: split_pairing(p)[1])
    order = ["prop10", "thr9_p10", "thr9_p5", "human_severity", NEVER, CLONE]
    focal = focal[focal["focal"].isin(order)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    titles = {
        CLONE: "rival: the clone (both seats punish)",
        NEVER: "rival: never-punish (the refuge)",
    }
    for ax, rival in zip(axes, RIVALS):
        sub = focal[focal["rival"] == rival]
        ax.axhline(4.0, color=MUTED, lw=1.0, ls="--", zorder=1)
        # fixed colour per manager, so a rule keeps its colour across panels;
        # the two rivals share slot 4 because only one of them is ever a
        # coloured focal in a panel -- the other is the grey control.
        slot = {"prop10": 0, "thr9_p10": 1, "thr9_p5": 2, "human_severity": 3}
        ends = []
        for name in [n for n in order if n in set(sub["focal"])]:
            s = sub[sub["focal"] == name].sort_values("round_number")
            is_ctrl = name == rival
            color = MUTED if is_ctrl else SERIES[slot.get(name, 4)]
            ax.plot(
                s["round_number"],
                s["group_size"],
                color=color,
                lw=2.4 if is_ctrl else 1.8,
                ls="--" if is_ctrl else "-",
                zorder=2,
            )
            ends.append(
                (
                    float(s["group_size"].iloc[-1]),
                    float(s["round_number"].iloc[-1]),
                    name + (" (control)" if is_ctrl else ""),
                    color,
                )
            )
        # direct labels instead of a legend, pushed apart so none collide
        ends.sort()
        min_gap = 0.30
        ys = [e[0] for e in ends]
        for j in range(1, len(ys)):
            ys[j] = max(ys[j], ys[j - 1] + min_gap)
        for (_, x, name, color), y in zip(ends, ys):
            ax.annotate(
                name,
                (x, y),
                xytext=(5, 0),
                textcoords="offset points",
                color=color,
                fontsize=8,
                va="center",
            )
        ax.set_title(titles[rival], fontsize=10, color=INK)
        ax.set_xlabel("round")
        ax.set_xlim(0, 30)
    axes[0].set_ylabel("members held by the focal seat")
    axes[0].set_ylim(0, 8)
    fig.suptitle(
        "Group size held by the focal manager, against two rivals "
        f"({tag}); dashed grey = symmetric control, dashed line = 4 (even split)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    p = os.path.join(out_dir, f"group_size_{tag}.jpg")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def fig_margins(wc, out_dir, tag):
    """Focal seat against its symmetric control, both rivals, common pool."""
    d = wc.copy()
    d["label"] = d["focal"] + " vs " + d["rival"]
    d = d.sort_values("d_pool_corr")
    fig, ax = plt.subplots(figsize=(8.0, 0.38 * len(d) + 2.0))
    colors = [SERIES[0] if r == CLONE else SERIES[1] for r in d["rival"]]
    y = np.arange(len(d))
    ax.barh(y, d["d_pool_corr"], color=colors, height=0.72, linewidth=0)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot(
            [r["d_pool_corr_lo"], r["d_pool_corr_hi"]],
            [i, i],
            color=INK,
            lw=1.2,
            solid_capstyle="butt",
        )
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=9)
    ax.set_xlabel("focal seat's common pool, minus the same seat of the control")
    ax.set_title(
        f"What the rule buys its own seat, by rival ({tag})\n"
        "blue = rival is the clone, orange = rival is never-punish",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    p = os.path.join(out_dir, f"margins_vs_control_{tag}.jpg")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def fig_size_vs_pool(summary, out_dir, tag):
    """Holding members and filling the pool are not the same thing."""
    d = summary[summary["seat"] == "focal"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for rival, color in zip(RIVALS, SERIES[:2]):
        sub = d[d["rival"] == rival]
        ax.scatter(
            sub["group_size_late"],
            sub["share_corr"],
            s=70,
            color=color,
            zorder=3,
            label=f"rival: {rival}",
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["manager"],
                (r["group_size_late"], r["share_corr"]),
                xytext=(6, 3),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
    ax.axvline(4.0, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.set_xlabel("members held, rounds 16-23")
    ax.set_ylabel("common pool per member (corrected accounting)")
    ax.set_title(
        f"Holding members against filling the pool, focal seat ({tag})",
        fontsize=10,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, f"size_vs_pool_{tag}.jpg")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


TRAJ = [
    ("group_size", "members held"),
    ("pool_corr", "common pool, undivided"),
    ("mean_c", "mean contribution"),
    ("mean_p", "mean punishment"),
    ("payoff_corr_pc", "contributor payoff per member"),
    ("share_corr", "pool per member"),
]


def fig_trajectories(series, out_dir, tag, rival):
    """Both seats of every pairing against this rival, round by round."""
    sub = series[series["pairing"].str.endswith(f"_vs_{rival}")].copy()
    sub["focal"] = sub["pairing"].map(lambda p: split_pairing(p)[0])
    order = [
        n
        for n in ["prop10", "thr9_p10", "thr9_p5", "human_severity", NEVER, CLONE]
        if n in set(sub["focal"])
    ]
    slot = {"prop10": 0, "thr9_p10": 1, "thr9_p5": 2, "human_severity": 3}

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6))
    for ax, (metric, label) in zip(axes.ravel(), TRAJ):
        for name in order:
            s = sub[(sub["focal"] == name) & (sub["seat"] == "focal")]
            s = s.sort_values("round_number")
            color = MUTED if name == rival else SERIES[slot.get(name, 4)]
            ax.plot(
                s["round_number"],
                s[metric],
                color=color,
                lw=1.7,
                ls="--" if name == rival else "-",
            )
        # the rival seat of the symmetric control: what the seat does with
        # no rule in it at all
        ctrl = series[
            (series["pairing"] == f"{rival}_vs_{rival}") & (series["seat"] == "rival")
        ].sort_values("round_number")
        ax.plot(ctrl["round_number"], ctrl[metric], color=INK, lw=1.0, ls=":")
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xlabel("round", fontsize=8)
        ax.tick_params(labelsize=8)
    handles = [
        plt.Line2D(
            [],
            [],
            color=MUTED if n == rival else SERIES[slot.get(n, 4)],
            ls="--" if n == rival else "-",
            lw=1.7,
            label=n,
        )
        for n in order
    ]
    handles.append(
        plt.Line2D(
            [], [], color=INK, ls=":", lw=1.0, label=f"{rival} seat of the control"
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(f"Focal seat against {rival}, per round ({tag})", fontsize=11, y=1.01)
    fig.tight_layout()
    p = os.path.join(out_dir, f"trajectories_vs_{rival}_{tag}.jpg")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------- #


def load(sim_dirs):
    frames = []
    for i, d in enumerate(sim_dirs):
        df = pd.read_parquet(os.path.join(d, "per_round.parquet"))
        df["pairing"] = df["run"].map(lambda s: RUN_RE.match(s).group("pairing"))
        # episodes are numbered 0..99 inside every run, so pooling seeds
        # needs them made distinct first
        df["episode"] = f"{i}_" + df["episode"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def sweep_side_by_side(summary, wc, out_dir):
    """The sweep's margin over the clone, and this setting's, side by side.

    The sweep's number is a whole-population total in self-play: both seats
    carried the manager, so it counts TWO groups. The paired margin changes
    only one seat, so the like-for-like sweep figure is halved -- that is the
    `_per_seat` column, and it is a first-order normalisation, not an
    identity (in self-play the rival group's behaviour changed too). What
    carries the claim is the sign and the size against each setting's own
    seed spread."""
    rows = []
    base = SWEEP_H2H_CG[CLONE]
    focal_level = summary[
        (summary["seat"] == "focal") & (summary["rival"] == CLONE)
    ].set_index("manager")["pool_corr"]
    for _, r in wc[wc["rival"] == CLONE].iterrows():
        if r["focal"] not in SWEEP_H2H_CG:
            continue
        margin = SWEEP_H2H_CG[r["focal"]] - base
        rows.append(
            {
                "manager": r["focal"],
                "sweep_cg_self_play": SWEEP_H2H_CG[r["focal"]],
                "sweep_cg_per_seat": SWEEP_H2H_CG[r["focal"]] / 2,
                "paired_focal_pool": float(focal_level.get(r["focal"], np.nan)),
                "sweep_margin_vs_clone": margin,
                "sweep_margin_per_seat": margin / 2,
                "paired_margin_vs_clone_seat": r["d_pool_corr"],
                "paired_lo": r["d_pool_corr_lo"],
                "paired_hi": r["d_pool_corr_hi"],
                "paired_margin_group_size": r["d_group_size"],
                "paired_group_size_lo": r["d_group_size_lo"],
                "paired_group_size_hi": r["d_group_size_hi"],
            }
        )
    out = pd.DataFrame(rows).sort_values("sweep_margin_vs_clone", ascending=False)
    p = os.path.join(out_dir, "sweep_side_by_side.csv")
    out.to_csv(p, index=False)
    print(f"\nwrote {p}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return out


def aggregate(tags, out_dir):
    """Seed-to-seed spread: the yardstick every margin here is judged against."""
    frames = []
    for t in tags:
        d = pd.read_csv(os.path.join(out_dir, f"summary_{t}.csv"))
        d["tag"] = t
        frames.append(d)
    allseeds = pd.concat(frames, ignore_index=True)
    out = []
    for metric in ["group_size", "group_size_late", "pool_corr", "share_corr"]:
        piv = allseeds.pivot_table(
            index=["pairing", "seat"], columns="tag", values=metric
        )
        piv["mean"] = piv[list(tags)].mean(axis=1)
        piv["spread"] = piv[list(tags)].max(axis=1) - piv[list(tags)].min(axis=1)
        piv["sd"] = piv[list(tags)].std(axis=1)
        piv["metric"] = metric
        out.append(piv.reset_index())
    combined = pd.concat(out, ignore_index=True)
    p = os.path.join(out_dir, "seed_spread.csv")
    combined.to_csv(p, index=False)
    print(f"wrote {p}\n")
    for metric in ["group_size_late", "pool_corr"]:
        sub = combined[combined["metric"] == metric]
        print(f"--- {metric} ---")
        print(
            sub.drop(columns="metric").to_string(
                index=False, float_format=lambda v: f"{v:.2f}"
            )
        )
        print(
            f"  sd across seeds: median {sub['sd'].median():.2f}, "
            f"max {sub['sd'].max():.2f}\n"
        )
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dirs", nargs="*")
    ap.add_argument("--tag", default="s42")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--aggregate", nargs="+")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"grid.linewidth": 0.5, "axes.edgecolor": GRID})

    if args.aggregate:
        aggregate(args.aggregate, args.out_dir)
        return

    df = load(args.sim_dirs)
    check_dispatch(df)
    g = group_rounds(df)
    pe = per_episode(g)
    pe_late = per_episode(g, late=True)

    summary = summarise(pe, pe_late, timeout_stats(df))
    summary.to_csv(os.path.join(args.out_dir, f"summary_{args.tag}.csv"), index=False)

    ww = within_world(pe)
    ww.to_csv(os.path.join(args.out_dir, f"within_world_{args.tag}.csv"), index=False)

    wc = vs_control(pe)
    wc.to_csv(os.path.join(args.out_dir, f"vs_control_{args.tag}.csv"), index=False)

    series = round_series(g)
    series.to_csv(
        os.path.join(args.out_dir, f"round_series_{args.tag}.csv"), index=False
    )

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
    print(f"=== per-seat summary ({args.tag}) ===")
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print(f"\n=== focal minus rival, same world, paired by episode ({args.tag}) ===")
    for _, r in ww.iterrows():
        print(
            f"  {r['focal']:>15s} vs {r['rival']:<12s} "
            f"size {r['d_group_size']:+6.2f} "
            f"[{r['d_group_size_lo']:+6.2f}, {r['d_group_size_hi']:+6.2f}]   "
            f"pool {r['d_pool_corr']:+7.2f} "
            f"[{r['d_pool_corr_lo']:+7.2f}, {r['d_pool_corr_hi']:+7.2f}]   "
            f"per-member {r['d_share_corr']:+6.2f} "
            f"[{r['d_share_corr_lo']:+6.2f}, {r['d_share_corr_hi']:+6.2f}]"
        )

    print(f"\n=== focal seat minus the same seat of the control ({args.tag}) ===")
    for _, r in wc.iterrows():
        print(
            f"  {r['focal']:>15s} vs {r['rival']:<12s} "
            f"size {r['d_group_size']:+6.2f} "
            f"[{r['d_group_size_lo']:+6.2f}, {r['d_group_size_hi']:+6.2f}]   "
            f"pool {r['d_pool_corr']:+7.2f} "
            f"[{r['d_pool_corr_lo']:+7.2f}, {r['d_pool_corr_hi']:+7.2f}]"
        )

    sweep_side_by_side(summary, wc, args.out_dir)

    fig_group_size(series, args.out_dir, args.tag)
    fig_margins(wc, args.out_dir, args.tag)
    fig_size_vs_pool(summary, args.out_dir, args.tag)
    for rival in RIVALS:
        fig_trajectories(series, args.out_dir, args.tag, rival)


if __name__ == "__main__":
    main()
