"""Diagnostic for the switch-herding-copula recalibration (step 3).

Why did an intra-group herding latent on the switch slot produce LESS
segregation (SC 2.187 -> 2.967) when static reasoning says it should raise
Var(out-flow) in both groups and hence E|net flow|?

Three frames, all read through the evaluation suite's own loaders so the
frame matches SC's definition exactly:

  human   experiments/2group_8agent_50ep.csv                 (50 episodes)
  B       the parent stack, PR #165, SC 2.1867905905576634   (100 episodes)
  C       PR #166's FAILED run, SC 2.967099851949342         (100 episodes)

C is the same stack as B with only the switch slot swapped to the herding
copula (rho = 0.116482333585783, phi = 0.70366020589033,
copula_switch_every = 4). It lives in a SIBLING WORKTREE and is read
READ-ONLY -- this script never writes there, and writes nothing at all
outside stdout.

Conventions. Membership is `group_id` (per-round). `does_switch` sits at the
DECISION round s in {3, 7, 11, 15, 19}; the change realises between s and
s+1, so out-flow at s counted from `does_switch` IS the realised flow.
Signed net flow is

    net(s) = out_1(s) - out_0(s) = size_0(s + 1) - size_0(s),

i.e. the change in label 0's size. Imbalance is |size_0 - 4| (equivalently
larger-group size - 4). SC's support is rounds >= 4.

Run: PYTHONPATH=src python scripts/data_analysis/switch_copula_recal_diagnostic.py
"""

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import norm, wasserstein_distance

from aimanager.evaluation_suite.convert import load_human, load_sim
from aimanager.evaluation_suite.metrics import _spread_ratio

HUMAN_CSV = "experiments/2group_8agent_50ep.csv"
PARENT_PARQUET = (
    "plots/simulation/"
    "23_2g8a_contr_herding_copula_v2_self_gnncopar1_contr_gnn_switch/"
    "per_round.parquet"
)
# PR #166's failed run, in its own worktree. READ-ONLY: never written to.
V166_PARQUET = (
    "/Users/ertuerkan/Desktop/algorithmic-institutions/.claude/worktrees/"
    "switch-herding-copula-v3/plots/simulation/"
    "23_2g8a_full_copula_v3_self_gnncopar1_contr_herdcopar1_switch/"
    "per_round.parquet"
)

DECISION_ROUNDS = [3, 7, 11, 15, 19]
BLOCKS = [(4, 7), (8, 11), (12, 15), (16, 19), (20, 23)]
SIZES = [4, 5, 6, 7, 8]
LABELS = ["human", "B parent", "C #166"]
GROUP_CELL = ["episode_id", "round_number", "group_id"]
BOOT_REPEATS = 400
BOOT_SEED = 0


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_frames():
    frames = {"human": load_human(HUMAN_CSV)}
    for label, path in [("B parent", PARENT_PARQUET), ("C #166", V166_PARQUET)]:
        runs = load_sim(path)
        assert len(runs) == 1, f"{label}: expected one run, got {list(runs)}"
        frames[label] = next(iter(runs.values()))
    return frames


def decision_table(df):
    """One row per (episode, decision round): sizes, per-label out-flow,
    signed net flow, gross flow, imbalance before / after."""
    dec = df[df["round_number"].isin(DECISION_ROUNDS)]
    agg = dec.groupby(GROUP_CELL).agg(
        n=("does_switch", "size"), out=("does_switch", "sum")
    )
    wide = agg.unstack("group_id")
    wide.columns = [f"{a}_{int(b)}" for a, b in wide.columns]
    for c in ["n_0", "n_1", "out_0", "out_1"]:
        if c not in wide:
            wide[c] = 0
    wide = wide.fillna(0).astype(int).reset_index()
    wide["net"] = wide["out_1"] - wide["out_0"]
    wide["gross"] = wide["out_0"] + wide["out_1"]
    wide["block"] = wide[["out_0", "out_1"]].max(axis=1)
    wide["size_0_after"] = wide["n_0"] + wide["net"]
    wide["imb_before"] = (wide["n_0"] - 4).abs()
    wide["imb_after"] = (wide["size_0_after"] - 4).abs()
    wide["larger_before"] = wide[["n_0", "n_1"]].max(axis=1)
    wide["larger_after"] = np.maximum(wide["size_0_after"], 8 - wide["size_0_after"])
    wide["signed_imb_before"] = wide["n_0"] - 4
    return wide.sort_values(["episode_id", "round_number"]).reset_index(drop=True)


def cell_table(df):
    """One row per (episode, decision round, group label): size n and the
    number k of that label's members who switched."""
    dec = df[df["round_number"].isin(DECISION_ROUNDS)]
    cells = (
        dec.groupby(GROUP_CELL)
        .agg(n=("does_switch", "size"), k=("does_switch", "sum"))
        .reset_index()
    )
    cells["k"] = cells["k"].astype(int)
    return cells


def emd_sizes(a, b):
    return wasserstein_distance(np.asarray(a, float), np.asarray(b, float))


def larger_sizes(df, lo=4, hi=23):
    """SC's own observation: larger-group size per (episode, round) over the
    round window [lo, hi]. An empty group vanishes from the groupby, so the
    max is 8 exactly when one group has emptied -- SC keeps those rounds."""
    sub = df[(df["round_number"] >= lo) & (df["round_number"] <= hi)]
    return sub.groupby(GROUP_CELL).size().groupby(["episode_id", "round_number"]).max()


def icc(cells):
    """Pairwise (Bernoulli) intra-cell correlation of switching:
    r = (E[k(k-1)]/E[n(n-1)] - p^2) / (p(1-p)), p the pooled switch rate.
    This is the observable counterpart of the copula's latent rho."""
    sub = cells[cells["n"] >= 2]
    p = cells["k"].sum() / cells["n"].sum()
    same = (sub["k"] * (sub["k"] - 1)).sum()
    pairs = (sub["n"] * (sub["n"] - 1)).sum()
    return p, (same / pairs - p * p) / (p * (1 - p))


def copula_gain(rho, p):
    """Bernoulli pairwise correlation a Gaussian copula of latent correlation
    rho adds at marginal p: Cov ~ rho * phi(t)^2 with t = Phi^-1(1 - p), so
    dr/drho ~ phi(t)^2 / (p(1-p)) to first order in rho."""
    t = norm.ppf(1.0 - p)
    return rho * norm.pdf(t) ** 2 / (p * (1.0 - p))


def group_means(df, measure="contribution"):
    v = df.dropna(subset=[measure])
    return v.groupby(GROUP_CELL)[measure].mean()


def validity_report(frames):
    rule("0. Frames and decision-row validity")
    for label, df in frames.items():
        dec = df[df["switch_mask"]]
        print(
            f"{label:>9}: {len(df):6d} rows, {df['episode_id'].nunique():3d} episodes, "
            f"{len(dec):5d} decision rows, {int(dec['does_switch'].sum()):4d} realised "
            f"switches, {int((~dec['switch_valid']).sum()):4d} invalid (timeout), "
            f"{int((dec['does_switch'] & ~dec['switch_valid']).sum())} of them realised"
        )
    print(
        "\n  does_switch is derived from the realised membership change, so "
        "out-flow\n  counted from it IS realised flow. Dropping the 109 human "
        "timeout rows would\n  drop 11 realised switches (0.6% of the human "
        "decision rows), so every flow\n  number below uses realised flow."
    )


def m1_per_round_table(tabs):
    rule("1. Per (episode, decision round) flow table -- head and means")
    cols = [
        "episode_id",
        "round_number",
        "n_0",
        "n_1",
        "out_0",
        "out_1",
        "net",
        "gross",
        "imb_before",
        "imb_after",
        "larger_before",
        "larger_after",
    ]
    for label, t in tabs.items():
        print(f"\n{label}: {len(t)} (episode, decision round) cells")
        print(t.head(3)[cols].to_string(index=False))
        print(
            f"  means: n_0 {t['n_0'].mean():.3f}  out_0 {t['out_0'].mean():.3f}  "
            f"out_1 {t['out_1'].mean():.3f}  net {t['net'].mean():+.4f}  "
            f"gross {t['gross'].mean():.3f}  max block {t['block'].mean():.3f}"
        )


def m2_net_and_gross(tabs):
    rule("2. |net flow| and gross flow per decision round (H1 vs H2)")
    rows = []
    for label, t in tabs.items():
        for key, sub in [(s, t[t["round_number"] == s]) for s in DECISION_ROUNDS] + [
            ("all", t)
        ]:
            rows.append(
                {
                    "frame": label,
                    "round": key,
                    "E|net|": sub["net"].abs().mean(),
                    "sd(net)": sub["net"].std(),
                    "E gross": sub["gross"].mean(),
                    "switch rate": sub["gross"].sum() / (8 * len(sub)),
                }
            )
    out = pd.DataFrame(rows).pivot(index="round", columns="frame")
    print(out.reindex(DECISION_ROUNDS + ["all"]).round(4).to_string())

    for name in ["net", "gross"]:
        print(f"\n  Distribution of {'|net flow|' if name == 'net' else 'gross flow'}:")
        dist = {}
        for label, t in tabs.items():
            v = t[name].abs() if name == "net" else t[name]
            dist[label] = v.value_counts(normalize=True).sort_index()
        print(pd.DataFrame(dist).fillna(0).round(4).to_string())

    print("\n  Var(net) = Var(out_0) + Var(out_1) - 2Cov:")
    rows = []
    for label, t in tabs.items():
        rows.append(
            {
                "frame": label,
                "Var(out_0)": t["out_0"].var(),
                "Var(out_1)": t["out_1"].var(),
                "Cov": t[["out_0", "out_1"]].cov().iloc[0, 1],
                "corr": t["out_0"].corr(t["out_1"]),
                "Var(net)": t["net"].var(),
                "Var(gross)": t["gross"].var(),
            }
        )
    print(pd.DataFrame(rows).set_index("frame").round(4).to_string())

    print("\n  E[|net| | gross] -- how directional a given amount of movement is:")
    d = {}
    for label, t in tabs.items():
        d[label] = t.groupby("gross")["net"].apply(lambda x: x.abs().mean())
    d["n human"] = tabs["human"]["gross"].value_counts().sort_index()
    print(pd.DataFrame(d).round(3).to_string())


def m3_coswitching(cell_tabs):
    rule("3. Co-switching within a group label (is the copula working?)")
    print("  Pooled switch rate p and intra-cell pairwise correlation r:")
    for label, cells in cell_tabs.items():
        p, r = icc(cells)
        print(f"    {label:>9}: p = {p:.4f}   r = {r:+.4f}")

    print("\n  r per decision round (where the human co-switching lives):")
    rows = {}
    for label, cells in cell_tabs.items():
        rows[label] = cells.groupby("round_number").apply(lambda g: icc(g)[1])
    print(pd.DataFrame(rows).round(4).to_string())

    print("\n  Round 3 only (both labels are exactly 4 -- the founding exodus):")
    for label, cells in cell_tabs.items():
        s = cells[(cells["round_number"] == 3) & (cells["n"] == 4)]
        p = s["k"].sum() / (4 * len(s))
        share = s["k"].value_counts(normalize=True).reindex(range(5)).fillna(0)
        print(
            f"    {label:>9}: {len(s):3d} cells  p {p:.4f}  Var(k) {s['k'].var():.4f}  "
            f"binom {4 * p * (1 - p):.4f}  k=  "
            + "  ".join(f"{i}:{share[i]:.3f}" for i in range(5))
        )

    print("\n  Distribution of k switchers, by group size n (share within n):")
    for label, cells in cell_tabs.items():
        sub = cells[cells["n"] >= 2]
        tab = sub.groupby("n")["k"].value_counts(normalize=True).unstack("k").fillna(0)
        counts = sub.groupby("n").size().rename("cells")
        rate = (sub.groupby("n")["k"].sum() / sub.groupby("n")["n"].sum()).rename("p_n")
        print(f"\n    {label}:")
        print(pd.concat([counts, rate.round(4), tab.round(3)], axis=1).to_string())

    print("\n  Overdispersion of k vs binomial(n, p_n):")
    rows = []
    for label, cells in cell_tabs.items():
        sub = cells[cells["n"] >= 2]
        for n, grp in sub.groupby("n"):
            p_n = grp["k"].sum() / (n * len(grp))
            binom = n * p_n * (1 - p_n)
            rows.append(
                {
                    "frame": label,
                    "n": n,
                    "Var(k)": grp["k"].var(),
                    "binom Var": binom,
                    "ratio": grp["k"].var() / binom if binom > 0 else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    print(
        df.pivot(index="n", columns="frame", values=["Var(k)", "binom Var", "ratio"])
        .round(3)
        .to_string()
    )

    print("\n  Dose implied by the round-3 gap, to first order in rho:")
    r3 = {label: icc(c[c["round_number"] == 3]) for label, c in cell_tabs.items()}
    p_h, r_h = r3["human"]
    p_b, r_b = r3["B parent"]
    need = (r_h - r_b) * p_b * (1 - p_b) / norm.pdf(norm.ppf(1 - p_b)) ** 2
    print(
        f"    human r {r_h:.4f} vs B {r_b:.4f} -- gap {r_h - r_b:.4f} "
        f"at p ~ {p_b:.3f}.\n"
        f"    d r / d rho ~ {copula_gain(1.0, p_b):.3f}, so closing it needs "
        f"rho ~ {need:.3f}.\n"
        f"    #166's rho = 0.116482 buys only "
        f"{copula_gain(0.116482333585783, p_b):.4f} of it "
        f"({100 * copula_gain(0.116482333585783, p_b) / (r_h - r_b):.0f}%)."
    )


def m4_directional_consistency(tabs):
    rule("4. Directional consistency of net flow within an episode (H2)")
    rows = []
    for label, t in tabs.items():
        arr = (
            t.pivot(index="episode_id", columns="round_number", values="net")
            .reindex(columns=DECISION_ROUNDS)
            .to_numpy(dtype=float)
        )
        sgn = np.sign(arr)
        a, b = sgn[:, :-1].ravel(), sgn[:, 1:].ravel()
        both = (a != 0) & (b != 0) & ~np.isnan(a) & ~np.isnan(b)
        x, y = arr[:, :-1].ravel(), arr[:, 1:].ravel()
        ok = ~np.isnan(x) & ~np.isnan(y)
        nz = np.where(sgn == 0, np.nan, sgn)
        with np.errstate(invalid="ignore"):
            one_signed = np.nanmax(nz, axis=1) == np.nanmin(nz, axis=1)
        tot = np.nansum(arr, axis=1)
        abstot = np.nansum(np.abs(arr), axis=1)
        rows.append(
            {
                "frame": label,
                "P(same sign | both != 0)": float((a[both] == b[both]).mean()),
                "lag-1 corr(net)": float(np.corrcoef(x[ok], y[ok])[0, 1]),
                "share episodes one-signed": float(
                    np.nanmean(one_signed.astype(float))
                ),
                "E|sum net| (= E|final imb|)": np.abs(tot).mean(),
                "E sum|net|": abstot.mean(),
                "cancellation index": np.abs(tot).mean() / abstot.mean(),
            }
        )
    print(pd.DataFrame(rows).set_index("frame").round(4).to_string())
    print(
        "\n  cancellation index = E|sum of net flows| / E[sum of |net flows|]:\n"
        "  1.0 = every wave pushes the same way, 0 = perfect cancellation."
    )

    print("\n  Sign sequence of net flow over the 5 decision rounds, top patterns:")
    for label, t in tabs.items():
        piv = (
            t.pivot(index="episode_id", columns="round_number", values="net")
            .reindex(columns=DECISION_ROUNDS)
            .fillna(0)
        )
        seq = piv.apply(
            lambda r: "".join("+" if v > 0 else "-" if v < 0 else "0" for v in r),
            axis=1,
        )
        top = seq.value_counts(normalize=True).head(5).round(3)
        print(f"    {label:>9}: " + ", ".join(f"{k} {v:.3f}" for k, v in top.items()))


def m5_mean_reversion(tabs):
    rule("5. Mean reversion of imbalance across a decision round (H1)")
    print("  Transition matrix P(imb_after | imb_before), rows = imb_before:")
    for label, t in tabs.items():
        mat = (
            pd.crosstab(t["imb_before"], t["imb_after"], normalize="index")
            .reindex(index=range(5), columns=range(5))
            .fillna(0)
        )
        counts = t["imb_before"].value_counts().reindex(range(5)).fillna(0).astype(int)
        print(f"\n    {label} (n per row: {list(counts)}):")
        print(mat.round(3).to_string())

    print("\n  OLS, pooled over decision rounds:")
    rows = []
    for label, t in tabs.items():
        x, y = t["imb_before"].to_numpy(float), t["imb_after"].to_numpy(float)
        b1, b0 = np.polyfit(x, y, 1)
        c1, c0 = np.polyfit(
            t["signed_imb_before"].to_numpy(float), t["net"].to_numpy(float), 1
        )
        rows.append(
            {
                "frame": label,
                "imb_after ~ imb_before: slope": b1,
                "intercept": b0,
                "net ~ signed imb: slope": c1,
                "intercept ": c0,
                "E[imb_after] - E[imb_before]": y.mean() - x.mean(),
            }
        )
    print(pd.DataFrame(rows).set_index("frame").round(4).to_string())
    print(
        "\n  net ~ (size_0 - 4) with slope < 0 is a RESTORING force (the fuller\n"
        "  label exports more); -1 would be full equilibration to 4-4."
    )

    for col in ["net", "gross"]:
        print(f"\n  E[{col} | signed imbalance before]:")
        print(
            pd.DataFrame(
                {
                    lb: t.groupby("signed_imb_before")[col].mean()
                    for lb, t in tabs.items()
                }
            )
            .round(3)
            .to_string()
        )


def m6_sc_by_block(frames):
    rule("6. SC decomposition by round block (H3)")
    full = {label: larger_sizes(df) for label, df in frames.items()}
    tab = pd.DataFrame(
        {
            lb: s.value_counts(normalize=True).reindex(SIZES).fillna(0)
            for lb, s in full.items()
        }
    )
    tab.loc["mean"] = [full[c].mean() for c in tab.columns]
    tab.loc["n"] = [len(full[c]) for c in tab.columns]
    print("  Larger-group size shares over the whole SC support (rounds 4-23):")
    print(tab.round(4).to_string())
    print(
        f"\n  EMD vs human, whole support: "
        f"B {emd_sizes(full['B parent'], full['human']):.4f}   "
        f"C {emd_sizes(full['C #166'], full['human']):.4f}   "
        f"C vs B {emd_sizes(full['C #166'], full['B parent']):.4f}"
    )

    rows = []
    for lo, hi in BLOCKS:
        blk = {label: larger_sizes(df, lo, hi) for label, df in frames.items()}
        for label in LABELS:
            share = blk[label].value_counts(normalize=True)
            r = {"block": f"{lo}-{hi}", "frame": label, "mean": blk[label].mean()}
            r.update({f"p{sz}": share.get(sz, 0.0) for sz in SIZES})
            r["EMD vs human"] = (
                emd_sizes(blk[label], blk["human"]) if label != "human" else 0.0
            )
            rows.append(r)
    df = pd.DataFrame(rows).fillna(0.0)
    print("\n  Per block:")
    print(df.set_index(["block", "frame"]).round(4).to_string())

    print("\n  The missing right tail, p7 + p8 per block:")
    piv = df.assign(tail=df["p7"] + df["p8"]).pivot(
        index="block", columns="frame", values="tail"
    )
    piv = piv.reindex([f"{lo}-{hi}" for lo, hi in BLOCKS])
    piv["B - human"] = piv["B parent"] - piv["human"]
    piv["C - B"] = piv["C #166"] - piv["B parent"]
    print(piv.round(4).to_string())


def m7_absorption(tabs):
    rule("7. Absorption at the extreme")
    for state in [8, 7]:
        print(f"\n  Conditional on larger-group size = {state} at a decision round:")
        rows = []
        for label, t in tabs.items():
            sub = t[t["larger_before"] == state]
            rows.append(
                {
                    "frame": label,
                    "n": len(sub),
                    "share of cells": len(sub) / len(t),
                    "P(larger_after >= state)": (sub["larger_after"] >= state).mean(),
                    "P(larger_after == state)": (sub["larger_after"] == state).mean(),
                    "E larger_after": sub["larger_after"].mean(),
                    "E gross": sub["gross"].mean(),
                }
            )
        print(pd.DataFrame(rows).set_index("frame").round(4).to_string())

    print("\n  P(larger_after = 8 | larger_before):")
    print(
        pd.DataFrame(
            {
                lb: t.groupby("larger_before")
                .apply(lambda g: (g["larger_after"] == 8).mean())
                .rename(lb)
                for lb, t in tabs.items()
            }
        )
        .round(4)
        .to_string()
    )
    print("\n  P(larger_after = 8) by decision round:")
    print(
        pd.DataFrame(
            {
                lb: t.groupby("round_number")
                .apply(lambda g: (g["larger_after"] == 8).mean())
                .rename(lb)
                for lb, t in tabs.items()
            }
        )
        .round(4)
        .to_string()
    )


def m8_cross_slot_feedback(frames, tabs):
    rule("8. The cross-slot channel: sorting pressure and group differentiation")
    print("  Group-mean spread ratios (the CG / PD statistic itself; the")
    print("  independent-sampling floor is 1/sqrt(4) = 0.5):")
    rows = []
    for label, df in frames.items():
        v = df.dropna(subset=["contribution"])
        rows.append(
            {
                "frame": label,
                "CG ratio": float(_spread_ratio(df, "contribution", "CG").iloc[0]),
                "PD ratio": float(_spread_ratio(df, "punishment", "PD").iloc[0]),
                "sd(group mean contr)": group_means(df).std(),
                "sd(individual contr)": v["contribution"].std(),
            }
        )
    print(pd.DataFrame(rows).set_index("frame").round(4).to_string())

    print("\n  Between-group contribution gap and how flow follows it:")
    rows = []
    for label, df in frames.items():
        gap = group_means(df).unstack("group_id")
        gap = (gap[0] - gap[1]).rename("gap").reset_index()
        m = (
            tabs[label]
            .merge(gap, on=["episode_id", "round_number"])
            .dropna(subset=["gap"])
        )
        allr = gap[gap["round_number"] >= 4].dropna(subset=["gap"])
        nz = m[m["net"] != 0]
        rows.append(
            {
                "frame": label,
                "E|gap| rounds>=4": allr["gap"].abs().mean(),
                "E|gap| at decisions": m["gap"].abs().mean(),
                "corr(net, gap)": m["net"].corr(m["gap"]),
                "share net toward better group": (
                    np.sign(nz["net"]) == np.sign(nz["gap"])
                ).mean(),
            }
        )
    print(pd.DataFrame(rows).set_index("frame").round(4).to_string())
    print(
        "\n  net > 0 means label 0 grows; gap > 0 means label 0 contributes more,\n"
        "  so a positive correlation is flow toward the better-contributing group."
    )

    print("\n  Who moves -- assortative selection of the switchers:")
    rows = []
    for label, df in frames.items():
        gm = group_means(df).rename("gmean").reset_index()
        dec = df[df["round_number"].isin(DECISION_ROUNDS)].dropna(
            subset=["contribution"]
        )
        sw = dec[dec["does_switch"]][GROUP_CELL + ["contribution"]].copy()
        origin = sw.merge(gm, on=GROUP_CELL)
        sw["dest"] = 1 - sw["group_id"]
        dest = sw.merge(
            gm.rename(columns={"group_id": "dest"}),
            on=["episode_id", "round_number", "dest"],
        )
        rows.append(
            {
                "frame": label,
                "E[contr | switcher]": dec[dec["does_switch"]]["contribution"].mean(),
                "E[contr | stayer]": dec[~dec["does_switch"]]["contribution"].mean(),
                "corr(switch, own contr)": np.corrcoef(
                    dec["does_switch"].astype(float), dec["contribution"]
                )[0, 1],
                "corr(own contr, dest mean)": dest[["contribution", "gmean"]]
                .corr()
                .iloc[0, 1],
                "E[dest mean - origin mean]": dest["gmean"].mean()
                - origin["gmean"].mean(),
            }
        )
    print(pd.DataFrame(rows).set_index("frame").round(4).to_string())

    print("\n  Partition stability (mixing) over rounds >= 4:")
    rows = []
    for label, df in frames.items():
        sub = df[df["round_number"] >= 4]
        piv = sub.pivot_table(
            index=["episode_id", "round_number"],
            columns="participant_code",
            values="group_id",
        )
        co = []
        for _, g in piv.groupby(level=0):
            cols = [c for c in g.columns if g[c].notna().any()]
            a = g[cols].to_numpy()
            co += [
                (a[:, i] == a[:, j]).mean()
                for i, j in itertools.combinations(range(len(cols)), 2)
            ]
        co = np.asarray(co)
        t = tabs[label]
        ep = t.groupby("episode_id")[["out_0", "out_1", "gross"]].sum()
        ep = ep[ep["gross"] > 0]
        dec = df[df["switch_mask"]]
        per_player = dec.groupby(["episode_id", "participant_code"])[
            "does_switch"
        ].sum()
        top2 = per_player.groupby("episode_id").apply(
            lambda s: s.nlargest(2).sum() / max(s.sum(), 1)
        )
        rows.append(
            {
                "frame": label,
                "mean pair co-membership": co.mean(),
                "share pairs > 0.8": (co > 0.8).mean(),
                "exporter concentration": (
                    (ep["out_0"] - ep["out_1"]).abs() / ep["gross"]
                ).mean(),
                "top-2 movers' share of switches": top2.mean(),
                "switches per participant": per_player.mean(),
            }
        )
    print(pd.DataFrame(rows).set_index("frame").round(4).to_string())
    print(
        "\n  exporter concentration = per-episode |out_0 - out_1| / gross: how far\n"
        "  one label acts as THE exporter for the whole episode."
    )


def _boot_stats(df, human_sizes):
    t = decision_table(df)
    cells = cell_table(df)
    ls = larger_sizes(df)
    v = df.dropna(subset=["contribution"])
    return {
        "SC EMD": emd_sizes(ls.to_numpy(), human_sizes),
        "p8": float((ls == 8).mean()),
        "p7+p8": float((ls >= 7).mean()),
        "E|net| r3": t[t["round_number"] == 3]["net"].abs().mean(),
        "Var(k) r3": cells[cells["round_number"] == 3]["k"].var(),
        "CG ratio": group_means(df).std() / v["contribution"].std(),
    }


def m9_bootstrap(frames):
    rule("9. Episode bootstrap: is C's regression bigger than sampling noise?")
    human_sizes = larger_sizes(frames["human"]).to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    point, draws = {}, {}
    for label in ["B parent", "C #166"]:
        df = frames[label]
        by_ep = {e: g for e, g in df.groupby("episode_id")}
        eps = np.asarray(list(by_ep))
        point[label] = _boot_stats(df, human_sizes)
        out = []
        for _ in range(BOOT_REPEATS):
            parts = []
            for i, e in enumerate(rng.choice(eps, len(eps), replace=True)):
                g = by_ep[e].copy()
                g["episode_id"] = i
                parts.append(g)
            out.append(_boot_stats(pd.concat(parts, ignore_index=True), human_sizes))
        draws[label] = pd.DataFrame(out)
        print(f"\n  {label} ({BOOT_REPEATS} episode resamples, seed {BOOT_SEED}):")
        for k, val in point[label].items():
            lo, hi = np.percentile(draws[label][k], [2.5, 97.5])
            print(
                f"    {k:<10} {val:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   "
                f"sd {draws[label][k].std():.4f}"
            )
    print("\n  C - B, against the two runs' combined resampling sd:")
    for k in point["B parent"]:
        d = point["C #166"][k] - point["B parent"][k]
        sd = math.sqrt(draws["B parent"][k].var() + draws["C #166"][k].var())
        print(f"    {k:<10} {d:+.4f}   sd {sd:.4f}   z {d / sd:+.2f}")


def main():
    pd.set_option("display.width", 200)
    frames = load_frames()
    tabs = {label: decision_table(df) for label, df in frames.items()}
    cell_tabs = {label: cell_table(df) for label, df in frames.items()}

    validity_report(frames)
    m1_per_round_table(tabs)
    m2_net_and_gross(tabs)
    m3_coswitching(cell_tabs)
    m4_directional_consistency(tabs)
    m5_mean_reversion(tabs)
    m6_sc_by_block(frames)
    m7_absorption(tabs)
    m8_cross_slot_feedback(frames, tabs)
    m9_bootstrap(frames)


if __name__ == "__main__":
    main()
