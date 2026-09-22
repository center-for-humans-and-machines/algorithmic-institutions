"""Before/after table for the punisher timeout feature (auto/punisher-timeout-feature).

Before = the parent's `_ceiling` sims (PR #192, both punisher families carrying
the ceiling indicator), after = the `_timeout` sims (the punisher also sees
`contribution_valid` and is served the recorded 0 for a timed-out player), both
scored with the 22-row suite. Writes to
plots/data_analysis/evaluation/punisher_timeout_feature/:

- before_after.csv / .md: 22 rows x (case, before / after / delta / band /
  seed_sd / in_seed_sd / legible), the mean and rows <= 1, and the RCE band
  slopes with the protected-row checks;
- the gate verdict for the frontier case (target RCC: a band upgrade; mean
  within 10% of the baseline; RCE protected).

Two things differ from the parent's script beyond the retargeting:

- every movement carries the run-to-run **seed noise floor** of PR #195
  (`auto/seed-spread-noise-floor`), which retrained the same contributor six
  ways and measured how far each row travels on the seed alone. `in_seed_sd`
  is |delta| in units of that row's seed sd and `legible` is False when the
  movement is smaller than one seed sd, i.e. not distinguishable from a
  retrain of the unchanged model. Ten rows additionally have a band boundary
  inside one seed sd and cannot be gated on a single run at all (`ungateable`).
- the protected-row magnitude clause is the **amended** one: a halved band
  slope fires only when the candidate's slope is *not* closer to the human
  value and the change exceeds one pooled standard error. The parent's log
  (section "RCE band slopes") showed the raw relative-magnitude rule firing on
  a slope crossing zero toward the human and on changes of 0.49-0.90 pooled
  SE; both readings are kept side by side so the raw clause stays visible.

Usage (repo root, PYTHONPATH=src):
    python scripts/data_analysis/punisher_timeout_table.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from curpun_rebaseline import (  # noqa: E402
    MAIN,
    METRIC_ORDER,
    RUN,
    SIM,
    SKIP,
    band,
    human_rce_fit,
    md_table,
    read_scores,
    summarise,
)

OUT_DIR = "plots/data_analysis/evaluation/punisher_timeout_feature"

# Per-row seed sd over the six same-config contributor retrains of PR #195,
# plots/data_analysis/evaluation/seed_spread_noise_floor/per_row.csv on
# auto/seed-spread-noise-floor. Not this experiment's own variance (this one
# changes the punisher, not the contributor) but the right scale for reading
# any movement in these rows.
SEED_SD = {
    "CA": 0.1878, "CB": 0.2153, "CC": 0.1333, "CD": 0.1878, "CE": 0.0581,
    "CF": 0.1413, "CG": 0.3014, "SA": 0.1618, "SB": 0.0456, "SC": 0.1355,
    "PA": 0.0404, "PB": 0.0231, "PC": 0.0359, "PD": 0.0593, "RCA": 0.1412,
    "RCB": 0.1425, "RCC": 0.1631, "RCD": 0.2698, "RCE": 0.1063, "RSA": 0.1555,
    "RPA": 0.0175, "RPB": 0.0283,
}  # fmt: skip
# Rows whose nearest band boundary sits inside one seed sd: a band change on
# one run says nothing about the model. RCE, the protected row, is one.
UNGATEABLE = ["CA", "CB", "CC", "CD", "CF", "CG", "SA", "SB", "RCE", "RSA"]
MEAN_SEED_SD = 0.0473  # seed sd of the 22-row mean (gate-2 margin is 0.1033)
ROWS_LE1_SEED_SD = 3.1623  # seed sd of the rows <= 1 count, range 6 to 14
RCE_SLOPE_SEED_SD = {"0-4": 0.0182, "5-9": 0.0223, "10-14": 0.0263, "15-19": 0.0564}
CASES = [
    ("frontier", "PR 181 stimulus skip x joint-exodus GNN switch, lin_multinomial copula",
     SKIP, "lin_multinomial_copula_self", True),
    ("ref_lin", "main gnn x gnn, lin_multinomial (no copula)", MAIN,
     "lin_multinomial_self", False),
    ("ref_gnn", "main gnn x gnn, gnn punisher", MAIN, "gnn_self", False),
]  # fmt: skip


def _slope_row(stage, fit):
    """slope +- se (n) per band, so a rule firing on a band can be read
    against that band's own sampling error (one seed, one sim)."""
    return {
        "stage": stage,
        **{
            k: f"{fit.loc[k, 'slope']:+.3f} +- {fit.loc[k, 'se']:.3f} (n {fit.loc[k, 'n']})"
            for k in fit.index
        },
    }


def rce_check(human_fit, before_dir, after_dir, run, before_score, after_score):
    from aimanager.evaluation_suite.convert import load_sim
    from aimanager.evaluation_suite.metrics import GROUPS

    full = {}
    for stage, d in (("before", before_dir), ("after", after_dir)):
        p = os.path.join(SIM, d, "per_round.parquet")
        if os.path.exists(p):
            full[stage] = GROUPS["R"]._rce_fit(load_sim(p)[RUN + run])
    fits = {k: v["slope"] for k, v in full.items()}
    rows = [_slope_row("human", human_fit)]
    for stage, f in full.items():
        rows.append(_slope_row(stage, f))
    if len(fits) < 2:
        return rows, {}
    b, a = fits["before"], fits["after"]
    se_b, se_a = full["before"]["se"], full["after"]["se"]
    hs = np.sign(human_fit["slope"])
    checks = {
        "band_downgrade": band(after_score) != band(before_score)
        and after_score > before_score,
        "sign_lost": [
            k for k in a.index if np.sign(a[k]) != hs[k] and np.sign(b[k]) == hs[k]
        ],
        "magnitude_halved": [k for k in a.index if abs(a[k]) <= 0.5 * abs(b[k])],
        "signs_before": "".join("+" if v > 0 else "-" for v in b),
        "signs_after": "".join("+" if v > 0 else "-" for v in a),
        # |change| in units of the pooled SE of the two slopes: a rule that
        # fires at < ~2 is inside one seed's sampling error.
        "change_in_se": {
            k: round(abs(a[k] - b[k]) / np.hypot(se_b[k], se_a[k]), 2) for k in a.index
        },
        # the seed floor of PR #195 for the same band, for reading the change
        # against a retrain rather than against a resample
        "change_in_seed_sd": {
            k: round(abs(a[k] - b[k]) / RCE_SLOPE_SEED_SD[k], 2)
            for k in a.index
            if k in RCE_SLOPE_SEED_SD
        },
    }
    # Amended magnitude clause: a halved slope only counts as erosion when the
    # candidate is not closer to the human value AND the change is larger than
    # one pooled SE. The raw clause above stays visible next to it.
    hv = human_fit["slope"]
    checks["magnitude_eroded"] = [
        k
        for k in checks["magnitude_halved"]
        if abs(a[k] - hv[k]) >= abs(b[k] - hv[k])
        and abs(a[k] - b[k]) > np.hypot(se_b[k], se_a[k])
    ]
    checks["magnitude_halved_but_toward_human"] = [
        k for k in checks["magnitude_halved"] if abs(a[k] - hv[k]) < abs(b[k] - hv[k])
    ]
    return rows, checks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    human_fit = human_rce_fit()
    wide, md, verdict = {}, [], None
    for case, label, src, run, gated in CASES:
        before = read_scores(
            os.path.join(SIM, src + "_ceiling", "evaluation/scores.csv"), run
        )
        after = read_scores(
            os.path.join(SIM, src + "_timeout", "evaluation/scores.csv"), run
        )
        if before is None or after is None:
            print(
                f"{case}: missing scores (before={before is not None}, after={after is not None})"
            )
            continue
        t = pd.DataFrame({"before": before, "after": after})
        t["delta"] = t["after"] - t["before"]
        t["band"] = [
            f"{band(b)} -> {band(a)}" if band(b) != band(a) else band(b)
            for b, a in zip(t.before, t.after)
        ]
        # the seed noise floor beside every movement (PR #195)
        t["seed_sd"] = [SEED_SD.get(r, np.nan) for r in t.index]
        t["in_seed_sd"] = (t["delta"].abs() / t["seed_sd"]).round(2)
        t["legible"] = t["in_seed_sd"] >= 1.0
        t["ungateable"] = [r in UNGATEABLE for r in t.index]
        sb, sa = summarise(before), summarise(after)
        wide[f"{case}_before"], wide[f"{case}_after"] = before, after
        d_mean = sa["mean"] - sb["mean"]
        d_rows = sa["rows <= 1"] - sb["rows <= 1"]
        t.loc["mean"] = [
            sb["mean"],
            sa["mean"],
            d_mean,
            "",
            MEAN_SEED_SD,
            round(abs(d_mean) / MEAN_SEED_SD, 2),
            abs(d_mean) >= MEAN_SEED_SD,
            False,
        ]
        t.loc["rows <= 1"] = [
            sb["rows <= 1"],
            sa["rows <= 1"],
            d_rows,
            "",
            ROWS_LE1_SEED_SD,
            round(abs(d_rows) / ROWS_LE1_SEED_SD, 2),
            abs(d_rows) >= ROWS_LE1_SEED_SD,
            True,
        ]
        t.index.name = "row"
        rce_rows, checks = rce_check(
            human_fit,
            src + "_ceiling",
            src + "_timeout",
            run,
            before["RCE"],
            after["RCE"],
        )
        md += [f"### {case}: {label}", "", md_table(t, "{:.4f}"), ""]
        rce = pd.DataFrame(rce_rows).set_index("stage")
        rce.index.name = "RCE slopes"
        md += [md_table(rce, "{:+.3f}"), "", f"protected-row checks: {checks}", ""]
        if gated:
            mean_ok = sa["mean"] <= 1.10 * sb["mean"]
            rcc_up = (
                band(after["RCC"]) != band(before["RCC"])
                and after["RCC"] < before["RCC"]
            )
            # amended magnitude clause; the raw halving is reported, not gated
            rce_ok = not (
                checks["band_downgrade"]
                or checks["sign_lost"]
                or checks["magnitude_eroded"]
            )
            d_rcc = abs(after["RCC"] - before["RCC"])
            verdict = dict(
                rcc_before=before["RCC"],
                rcc_after=after["RCC"],
                gate1_rcc_band_upgrade=rcc_up,
                rcc_move_in_seed_sd=round(d_rcc / SEED_SD["RCC"], 2),
                rcc_clears_seed_floor=d_rcc >= SEED_SD["RCC"],
                mean_before=sb["mean"],
                mean_after=sa["mean"],
                gate2_ceiling=1.10 * sb["mean"],
                gate2_mean_ok=mean_ok,
                mean_move_in_seed_sd=round(
                    abs(sa["mean"] - sb["mean"]) / MEAN_SEED_SD, 2
                ),
                rce_protected_ok=rce_ok,
                rce_ungateable_on_one_run=True,
                verdict="SUCCESS" if (rcc_up and mean_ok and rce_ok) else "FAIL",
            )
            md += ["verdict: " + ", ".join(f"{k}={v}" for k, v in verdict.items()), ""]
    pd.DataFrame(wide).reindex(METRIC_ORDER).to_csv(
        os.path.join(OUT_DIR, "before_after.csv")
    )
    with open(os.path.join(OUT_DIR, "before_after.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
