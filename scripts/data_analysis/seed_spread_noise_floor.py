"""Run-to-run noise floor of the 22-row evaluation (auto/seed-spread-noise-floor).

Six arms of the PR #192 frontier stack that differ in the trained contributor
and in nothing else: PR #188's five seed-ensemble members (the shipped
stimulus-skip config byte-for-byte, seeds 1-5, same frozen copula carried on)
and the shipped artifact every recent experiment baselined against. Same
switch model, same punisher, same protocol, same simulation seed and episode
count -- so the spread across the six is what one retrain of an accepted model
does to the evaluation, with the simulation draw held fixed.

Writes plots/data_analysis/evaluation/seed_spread_noise_floor/:

- per_row.csv / .md: the six scores per row, mean, sd, min, max, range, the
  bands they land in, and -- the actionable column -- whether a scoring band
  boundary (1 / 2 / 5) falls inside one seed sd of the mean, i.e. whether that
  row can be gated on a single run at all;
- aggregates.csv: the 22-row mean, the rows <= 1 count and the four RCE band
  slopes per arm, each with its across-arm sd and range;
- levels.csv: the simulated contribution level and the share of players
  giving nothing per arm, against the human values (PR #194's open question);
- verdicts.csv: the movements PRs #190 / #192 / #193 / #194 turned on, each
  divided by the seed sd of the same quantity.

Usage (repo root):
    PYTHONPATH=src python scripts/data_analysis/seed_spread_noise_floor.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from curpun_rebaseline import (  # noqa: E402
    METRIC_ORDER,
    RUN,
    SIM,
    band,
    md_table,
    read_scores,
)

OUT_DIR = "plots/data_analysis/evaluation/seed_spread_noise_floor"
STACK = "23_2g8a_contr_stimulus_skip_seed{k}_self_gnncopar1_contr_gnn_switch_ceiling"
SHIPPED = "23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling"
PAIRING = "lin_multinomial_copula_self"
ARMS = [(f"seed_{k}", STACK.format(k=k)) for k in range(1, 6)] + [("shipped", SHIPPED)]
ARM_NAMES = [a for a, _ in ARMS]
BOUNDS = [1.0, 2.0, 5.0]
BANDS = ["0-4", "5-9", "10-14", "15-19"]
BAND_ORDER = ["<= 1", "1-2", "2-5", "> 5"]

# Human reference, recomputed from the evaluation suite's own loader.
HUMAN = dict(
    mean_c=9.457188841201717,
    share_c0=0.09356223175965665,
    share_c20=0.13444206008583692,
    mean_p=1.7912542151637114,
)

# The movements the four most recent verdicts turned on, quoted from their PR
# bodies. `kind` selects the seed sd they are read against: a row score, the
# 22-row mean, the rows <= 1 count, one RCE band slope, or a level statistic.
VERDICTS = [
    ("#190", "SC (declared target, no band)", "row:SC", 1.427, 1.329),
    ("#190", "RCD (declared target, wrong way)", "row:RCD", 1.309, 1.689),
    ("#190", "RSA (largest regression)", "row:RSA", 1.070, 1.644),
    ("#190", "SB (band upgrade, not declared)", "row:SB", 1.006, 0.898),
    ("#190", "RCE 10-14 slope (protected row fired)", "slope:10-14", -0.058, -0.020),
    ("#192", "RCC (declared target, no band)", "row:RCC", 1.5298, 1.2969),
    ("#192", "22-row mean", "mean", 1.0357, 1.0331),
    ("#192", "rows <= 1", "rows_le1", 13.0, 14.0),
    ("#192", "RCB (worsened, flagged)", "row:RCB", 1.5454, 1.6591),
    ("#192", "CG (worsened, flagged)", "row:CG", 1.5535, 1.7588),
    ("#193", "RCC (declared target, wrong way)", "row:RCC", 1.2969, 1.4237),
    ("#193", "RPA (declared target, wrong way)", "row:RPA", 0.6620, 0.6932),
    ("#193", "RCE 10-14 slope (protected row fired)", "slope:10-14", -0.043, -0.007),
    ("#193", "22-row mean (best recorded)", "mean", 1.0331, 0.9970),
    ("#193", "CG (band upgrade, not declared)", "row:CG", 1.7588, 0.9655),
    ("#194", "RCC (declared target, 8.6% short)", "row:RCC", 1.2969, 1.0857),
    ("#194", "RCE (protected row, band drop)", "row:RCE", 0.8823, 1.1058),
    ("#194", "RCE 0-4 slope (clearest erosion)", "slope:0-4", 0.087, 0.048),
    ("#194", "RCE 5-9 slope", "slope:5-9", 0.038, 0.014),
    ("#194", "RCE 15-19 slope", "slope:15-19", -0.130, -0.037),
    ("#194", "22-row mean", "mean", 1.0331, 1.0983),
    ("#194", "mean contribution level", "level:mean_c", 9.324, 10.355),
    ("#194", "share giving nothing", "level:share_c0", 0.082, 0.054),
]


def spread(values):
    """mean / sd / min / max / range of one quantity over the six arms."""
    v = np.asarray(values, dtype=float)
    return dict(
        mean=v.mean(),
        sd=v.std(ddof=1),
        min=v.min(),
        max=v.max(),
        range=v.max() - v.min(),
    )


def load_scores():
    cols = {}
    for name, d in ARMS:
        s = read_scores(os.path.join(SIM, d, "evaluation", "scores.csv"), PAIRING)
        assert s is not None, f"{name}: no scores at {d}"
        assert s.notna().all(), f"{name}: missing rows {s[s.isna()].index.tolist()}"
        cols[name] = s
    return pd.DataFrame(cols).reindex(METRIC_ORDER)


def load_levels():
    """Simulated contribution level and punishment, per arm."""
    rows = []
    for name, d in ARMS:
        p = pd.read_parquet(os.path.join(SIM, d, "per_round.parquet"))
        p = p[p["run"] == RUN + PAIRING]
        c = p["contribution"]
        rows.append(
            dict(
                arm=name,
                mean_c=c.mean(),
                share_c0=(c == 0).mean(),
                share_c20=(c == 20).mean(),
                mean_p=p["punishment"].mean(),
                n=len(p),
            )
        )
    return pd.DataFrame(rows).set_index("arm")


def load_rce_slopes():
    """The four RCE band slopes, their within-run SEs and row counts, per arm."""
    from aimanager.evaluation_suite.convert import load_sim
    from aimanager.evaluation_suite.metrics import GROUPS

    slopes, ses, ns = {}, {}, {}
    for name, d in ARMS:
        fit = GROUPS["R"]._rce_fit(
            load_sim(os.path.join(SIM, d, "per_round.parquet"))[RUN + PAIRING]
        )
        slopes[name], ses[name], ns[name] = fit["slope"], fit["se"], fit["n"]
    return (
        pd.DataFrame(slopes).reindex(BANDS),
        pd.DataFrame(ses).reindex(BANDS),
        pd.DataFrame(ns).reindex(BANDS),
    )


def per_row_table(scores):
    out = scores.copy()
    st = scores[ARM_NAMES].apply(lambda r: pd.Series(spread(r.values)), axis=1)
    out = pd.concat([out, st], axis=1)
    out["bands"] = [
        "/".join(
            sorted({band(v) for v in scores.loc[m, ARM_NAMES]}, key=BAND_ORDER.index)
        )
        for m in scores.index
    ]
    out["straddles_band"] = ["/" in b for b in out["bands"]]
    out["bounds_in_1sd"] = [
        ",".join(str(b) for b in BOUNDS if abs(r["mean"] - b) <= r["sd"])
        for _, r in out.iterrows()
    ]
    out["sd_to_nearest_bound"] = [
        min(abs(r["mean"] - b) for b in BOUNDS) / r["sd"] for _, r in out.iterrows()
    ]
    out["gateable_on_one_run"] = out["sd_to_nearest_bound"] >= 1.0
    out.index.name = "row"
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scores = load_scores()
    rows = per_row_table(scores)
    rows.to_csv(os.path.join(OUT_DIR, "per_row.csv"))

    agg = {
        "mean_22": {a: scores[a].mean() for a in ARM_NAMES},
        "rows_le1": {a: float((scores[a] <= 1).sum()) for a in ARM_NAMES},
    }
    slopes, ses, ns = load_rce_slopes()
    for b in BANDS:
        agg[f"rce_slope_{b}"] = {a: slopes.loc[b, a] for a in ARM_NAMES}
    agg_df = pd.DataFrame(agg).T[ARM_NAMES]
    st = agg_df.apply(lambda r: pd.Series(spread(r.values)), axis=1)
    agg_df = pd.concat([agg_df, st], axis=1)
    # The RCE rows also carry the median within-run sampling SE, so the seed
    # spread can be read against the error bar the verdicts already quoted.
    agg_df["within_run_se"] = [
        (
            ses.loc[q[len("rce_slope_") :]].median()
            if q.startswith("rce_slope_")
            else np.nan
        )
        for q in agg_df.index
    ]
    agg_df.index.name = "quantity"
    agg_df.to_csv(os.path.join(OUT_DIR, "aggregates.csv"))
    ns.to_csv(os.path.join(OUT_DIR, "rce_band_n.csv"))
    ses.to_csv(os.path.join(OUT_DIR, "rce_band_se.csv"))

    levels = load_levels()
    lv = levels.copy()
    for k, v in HUMAN.items():
        lv.loc["human", k] = v
    keys = ["mean_c", "share_c0", "share_c20", "mean_p"]
    lst = pd.DataFrame({k: spread(levels[k].values) for k in keys}).T
    lst["human"] = [HUMAN[k] for k in keys]
    lv.to_csv(os.path.join(OUT_DIR, "levels.csv"))
    lst.to_csv(os.path.join(OUT_DIR, "levels_spread.csv"))

    sd_of = {
        "mean": agg_df.loc["mean_22", "sd"],
        "rows_le1": agg_df.loc["rows_le1", "sd"],
    }
    for b in BANDS:
        sd_of[f"slope:{b}"] = agg_df.loc[f"rce_slope_{b}", "sd"]
    for m in METRIC_ORDER:
        sd_of[f"row:{m}"] = rows.loc[m, "sd"]
    for k in ["mean_c", "share_c0"]:
        sd_of[f"level:{k}"] = lst.loc[k, "sd"]
    vd = []
    for pr, what, kind, before, after in VERDICTS:
        sd = sd_of[kind]
        vd.append(
            dict(
                pr=pr,
                movement=what,
                before=before,
                after=after,
                delta=after - before,
                seed_sd=sd,
                in_seed_sd=abs(after - before) / sd,
                inside_floor=abs(after - before) <= sd,
            )
        )
    vdf = pd.DataFrame(vd)
    vdf.to_csv(os.path.join(OUT_DIR, "verdicts.csv"), index=False)

    show = rows[ARM_NAMES + ["mean", "sd", "range", "bands", "sd_to_nearest_bound"]]
    show = show.copy()
    show["gateable"] = np.where(rows["gateable_on_one_run"], "yes", "**no**")
    md = [
        "# Seed-to-seed noise floor of the 22-row evaluation\n",
        "Six arms of the PR #192 frontier stack differing only in the trained "
        "contributor (PR #188's seeds 1-5 plus the shipped artifact); identical "
        "switch model, punisher, protocol, simulation seed and episode count.\n",
        "## Per row\n",
        "`sd_to_nearest_bound` is the distance from the six-arm mean to the "
        "nearest band boundary (1 / 2 / 5) in units of the seed sd. Below 1 the "
        "row's band is not decided by the model.\n",
        md_table(show, "{:.4f}") + "\n",
        "## Aggregates\n",
        md_table(agg_df, "{:.4f}") + "\n",
        "## Contribution level (PR #194's open question)\n",
        md_table(lv, "{:.4f}") + "\n",
        md_table(lst, "{:.4f}") + "\n",
        "## The four recent verdicts against the floor\n",
        md_table(vdf.set_index("pr"), "{:.4f}") + "\n",
    ]
    with open(os.path.join(OUT_DIR, "per_row.md"), "w") as f:
        f.write("\n".join(md))

    pd.set_option("display.width", 220)
    print(show.to_string())
    print()
    print(agg_df.to_string())
    print()
    print(lv.to_string())
    print()
    print(vdf.to_string())
    print(f"\n-> {OUT_DIR}/")


if __name__ == "__main__":
    main()
