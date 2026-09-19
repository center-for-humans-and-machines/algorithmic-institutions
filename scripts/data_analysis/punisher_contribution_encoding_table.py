"""Before/after table for auto/punisher-contribution-encoding.

Before = the parent's `_ceiling` sims (numeric c_t plus the maximum
indicator), after = the `_cbins` sims (c_t as a 21-level one-hot), both
scored with the 22-row suite. Writes to
plots/data_analysis/evaluation/punisher_contribution_encoding/:

- before_after.csv / .md: 22 rows x (case, before / after / delta / band), the
  mean and rows <= 1, and the RCE band slopes with their standard errors, row
  counts and the protected-row checks;
- the gate verdict for the frontier case: gate 1 is a band upgrade on EITHER
  declared target (RPA or RCC), gate 2 is the 22-row mean within 10% of the
  parent's, and RCE is protected.

The magnitude clause of the protected-row rule is the amended one. The
parent's version fired twice on the reference stacks, once on a slope that
had moved TOWARD the human value through zero, so a relative threshold alone
reads a thin band's noise as erosion. It now fires only when all three hold:
the after slope is at most half the before slope in magnitude, the after
slope is NOT closer to the human slope than the before slope was, and the
change exceeds one pooled standard error of the two slopes.

Usage (repo root, PYTHONPATH=src):
    python scripts/data_analysis/punisher_contribution_encoding_table.py
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

OUT_DIR = "plots/data_analysis/evaluation/punisher_contribution_encoding"
BEFORE_SUFFIX, AFTER_SUFFIX = "_ceiling", "_cbins"
TARGETS = ["RPA", "RCC"]
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
    h, hs = human_fit["slope"], np.sign(human_fit["slope"])
    # |change| in units of the pooled SE of the two slopes: a rule that
    # fires at < ~1 is inside one seed's sampling error.
    in_se = {k: abs(a[k] - b[k]) / np.hypot(se_b[k], se_a[k]) for k in a.index}
    checks = {
        "band_downgrade": band(after_score) != band(before_score)
        and after_score > before_score,
        "sign_lost": [k for k in a.index if np.sign(a[k]) != hs[k] and np.sign(b[k]) == hs[k]],
        # amended: halved AND not closer to the human AND beyond one pooled SE
        "magnitude_eroded": [
            k
            for k in a.index
            if abs(a[k]) <= 0.5 * abs(b[k])
            and abs(a[k] - h[k]) >= abs(b[k] - h[k])
            and in_se[k] > 1.0
        ],
        "magnitude_halved_raw": [k for k in a.index if abs(a[k]) <= 0.5 * abs(b[k])],
        "signs_before": "".join("+" if v > 0 else "-" for v in b),
        "signs_after": "".join("+" if v > 0 else "-" for v in a),
        "change_in_se": {k: round(v, 2) for k, v in in_se.items()},
        "closer_to_human": {
            k: bool(abs(a[k] - h[k]) < abs(b[k] - h[k])) for k in a.index
        },
    }
    return rows, checks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    human_fit = human_rce_fit()
    wide, md, verdict = {}, [], None
    for case, label, src, run, gated in CASES:
        before = read_scores(os.path.join(SIM, src + BEFORE_SUFFIX, "evaluation/scores.csv"), run)
        after = read_scores(os.path.join(SIM, src + AFTER_SUFFIX, "evaluation/scores.csv"), run)
        if before is None or after is None:
            print(f"{case}: missing scores (before={before is not None}, after={after is not None})")
            continue
        t = pd.DataFrame({"before": before, "after": after})
        t["delta"] = t["after"] - t["before"]
        t["band"] = [f"{band(b)} -> {band(a)}" if band(b) != band(a) else band(b) for b, a in zip(t.before, t.after)]
        sb, sa = summarise(before), summarise(after)
        wide[f"{case}_before"], wide[f"{case}_after"] = before, after
        t.loc["mean"] = [sb["mean"], sa["mean"], sa["mean"] - sb["mean"], ""]
        t.loc["rows <= 1"] = [sb["rows <= 1"], sa["rows <= 1"], sa["rows <= 1"] - sb["rows <= 1"], ""]
        t.index.name = "row"
        rce_rows, checks = rce_check(
            human_fit, src + BEFORE_SUFFIX, src + AFTER_SUFFIX, run, before["RCE"], after["RCE"]
        )
        md += [f"### {case}: {label}", "", md_table(t, "{:.4f}"), ""]
        rce = pd.DataFrame(rce_rows).set_index("stage")
        rce.index.name = "RCE slopes"
        md += [md_table(rce, "{:+.3f}"), "", f"protected-row checks: {checks}", ""]
        if gated:
            mean_ok = sa["mean"] <= 1.10 * sb["mean"]
            upgrades = {
                m: band(after[m]) != band(before[m]) and after[m] < before[m]
                for m in TARGETS
            }
            rce_ok = not (
                checks["band_downgrade"]
                or checks["sign_lost"]
                or checks["magnitude_eroded"]
            )
            verdict = dict(
                **{f"{m}_before": before[m] for m in TARGETS},
                **{f"{m}_after": after[m] for m in TARGETS},
                gate1_band_upgrade=upgrades,
                gate1_ok=any(upgrades.values()),
                mean_before=sb["mean"], mean_after=sa["mean"],
                gate2_ceiling=1.10 * sb["mean"], gate2_mean_ok=mean_ok,
                rce_protected_ok=rce_ok,
                verdict="SUCCESS"
                if (any(upgrades.values()) and mean_ok and rce_ok)
                else "FAIL",
            )
            md += ["verdict: " + ", ".join(f"{k}={v}" for k, v in verdict.items()), ""]
    pd.DataFrame(wide).reindex(METRIC_ORDER).to_csv(os.path.join(OUT_DIR, "before_after.csv"))
    with open(os.path.join(OUT_DIR, "before_after.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
