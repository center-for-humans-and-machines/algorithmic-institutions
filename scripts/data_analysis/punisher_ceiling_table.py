"""Before/after table for the punisher ceiling fix (auto/punisher-ceiling-fix).

Before = the parent's `_curpun` sims (punisher on the current contribution),
after = the `_ceiling` sims (plus the full-contribution indicator), both
scored with the 22-row suite. Writes to
plots/data_analysis/evaluation/punisher_ceiling_fix/:

- before_after.csv / .md: 22 rows x (case, before / after / delta / band), the
  mean and rows <= 1, and the RCE band slopes with the protected-row checks
  (band downgrade, a slope losing the human sign, a slope magnitude halving);
- the gate verdict for the frontier case (target RCC: a band upgrade; mean
  within 10% of the baseline; RCE protected).

Usage (repo root, PYTHONPATH=src):
    python scripts/data_analysis/punisher_ceiling_table.py
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

OUT_DIR = "plots/data_analysis/evaluation/punisher_ceiling_fix"
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
    }
    return rows, checks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    human_fit = human_rce_fit()
    wide, md, verdict = {}, [], None
    for case, label, src, run, gated in CASES:
        before = read_scores(
            os.path.join(SIM, src + "_curpun", "evaluation/scores.csv"), run
        )
        after = read_scores(
            os.path.join(SIM, src + "_ceiling", "evaluation/scores.csv"), run
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
        sb, sa = summarise(before), summarise(after)
        wide[f"{case}_before"], wide[f"{case}_after"] = before, after
        t.loc["mean"] = [sb["mean"], sa["mean"], sa["mean"] - sb["mean"], ""]
        t.loc["rows <= 1"] = [
            sb["rows <= 1"],
            sa["rows <= 1"],
            sa["rows <= 1"] - sb["rows <= 1"],
            "",
        ]
        t.index.name = "row"
        rce_rows, checks = rce_check(
            human_fit,
            src + "_curpun",
            src + "_ceiling",
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
            rce_ok = not (
                checks["band_downgrade"]
                or checks["sign_lost"]
                or checks["magnitude_halved"]
            )
            verdict = dict(
                rcc_before=before["RCC"],
                rcc_after=after["RCC"],
                gate1_rcc_band_upgrade=rcc_up,
                mean_before=sb["mean"],
                mean_after=sa["mean"],
                gate2_ceiling=1.10 * sb["mean"],
                gate2_mean_ok=mean_ok,
                rce_protected_ok=rce_ok,
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
