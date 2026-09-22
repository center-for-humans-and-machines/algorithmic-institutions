"""Before/after table for the simulation timeout serving fix
(auto/sim-timeout-imputation).

Before = PR #194's `_timeout` sims, after = the `_simtimeout` sims. The two
differ in *nothing* but `environment.served_state` / `linear_ah`: same
artifacts, same seed, same episode count, nothing retrained. That makes this
comparison cleaner than a retrained candidate's -- the run-to-run variation
of an unchanged pair of artifacts under a fixed seed is zero, not the 0.138
of a retrain -- but the PR #195 seed floor is still the right scale for
reading each row, because it is the scale on which the *model* is uncertain.

Writes to plots/data_analysis/evaluation/sim_timeout_imputation/:

- before_after.csv / .md: 22 rows x (case, before / after / delta / band /
  seed_sd / in_seed_sd / legible / ungateable), the mean and rows <= 1, and
  the RCE band slopes with the protected-row checks;
- the gate verdict for the frontier case (declared targets SA / SB / SC / CG:
  a band upgrade on one of them; mean within 10% of the baseline; RCE
  protected under the amended magnitude clause).

Everything about the seed floor and the amended protected-row clause is
carried over verbatim from punisher_timeout_table.py; only the two sim
suffixes and the declared targets differ.

Usage (repo root, PYTHONPATH=src):
    python scripts/data_analysis/sim_timeout_table.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from curpun_rebaseline import (  # noqa: E402
    MAIN,
    METRIC_ORDER,
    SIM,
    SKIP,
    band,
    human_rce_fit,
    md_table,
    read_scores,
    summarise,
)
from punisher_timeout_table import (  # noqa: E402
    MEAN_SEED_SD,
    ROWS_LE1_SEED_SD,
    SEED_SD,
    UNGATEABLE,
    rce_check,
)

OUT_DIR = "plots/data_analysis/evaluation/sim_timeout_imputation"
BEFORE, AFTER = "_timeout", "_simtimeout"
# Declared in the log before the first simulation: the switch model's own
# decision (S family) and the contributor's group-level dispersion (CG).
TARGETS = ["SA", "SB", "SC", "CG"]
CASES = [
    ("frontier", "PR 181 stimulus skip x joint-exodus GNN switch, lin_multinomial copula",
     SKIP, "lin_multinomial_copula_self", True),
    ("ref_lin", "main gnn x gnn, lin_multinomial (no copula)", MAIN,
     "lin_multinomial_self", False),
    ("ref_gnn", "main gnn x gnn, gnn punisher", MAIN, "gnn_self", False),
]  # fmt: skip


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    human_fit = human_rce_fit()
    wide, md, verdict = {}, [], None
    for case, label, src, run, gated in CASES:
        before = read_scores(os.path.join(SIM, src + BEFORE, "evaluation/scores.csv"), run)
        after = read_scores(os.path.join(SIM, src + AFTER, "evaluation/scores.csv"), run)
        if before is None or after is None:
            print(f"{case}: missing scores (before={before is not None}, after={after is not None})")
            continue
        t = pd.DataFrame({"before": before, "after": after})
        t["delta"] = t["after"] - t["before"]
        t["band"] = [f"{band(b)} -> {band(a)}" if band(b) != band(a) else band(b) for b, a in zip(t.before, t.after)]
        t["seed_sd"] = [SEED_SD.get(r, np.nan) for r in t.index]
        t["in_seed_sd"] = (t["delta"].abs() / t["seed_sd"]).round(2)
        t["legible"] = t["in_seed_sd"] >= 1.0
        t["ungateable"] = [r in UNGATEABLE for r in t.index]
        t["target"] = [r in TARGETS for r in t.index]
        sb, sa = summarise(before), summarise(after)
        wide[f"{case}_before"], wide[f"{case}_after"] = before, after
        d_mean = sa["mean"] - sb["mean"]
        d_rows = sa["rows <= 1"] - sb["rows <= 1"]
        t.loc["mean"] = [sb["mean"], sa["mean"], d_mean, "", MEAN_SEED_SD,
                         round(abs(d_mean) / MEAN_SEED_SD, 2),
                         abs(d_mean) >= MEAN_SEED_SD, False, False]
        t.loc["rows <= 1"] = [sb["rows <= 1"], sa["rows <= 1"], d_rows, "",
                              ROWS_LE1_SEED_SD,
                              round(abs(d_rows) / ROWS_LE1_SEED_SD, 2),
                              abs(d_rows) >= ROWS_LE1_SEED_SD, True, False]
        t.index.name = "row"
        rce_rows, checks = rce_check(
            human_fit, src + BEFORE, src + AFTER, run, before["RCE"], after["RCE"]
        )
        md += [f"### {case}: {label}", "", md_table(t, "{:.4f}"), ""]
        rce = pd.DataFrame(rce_rows).set_index("stage")
        rce.index.name = "RCE slopes"
        md += [md_table(rce, "{:+.3f}"), "", f"protected-row checks: {checks}", ""]
        if gated:
            mean_ok = sa["mean"] <= 1.10 * sb["mean"]
            upgrades = [
                r for r in TARGETS
                if band(after[r]) != band(before[r]) and after[r] < before[r]
            ]
            rce_ok = not (checks["band_downgrade"] or checks["sign_lost"] or checks["magnitude_eroded"])
            verdict = dict(
                targets=TARGETS,
                target_before={r: round(before[r], 4) for r in TARGETS},
                target_after={r: round(after[r], 4) for r in TARGETS},
                target_in_seed_sd={
                    r: round(abs(after[r] - before[r]) / SEED_SD[r], 2) for r in TARGETS
                },
                gate1_band_upgrades=upgrades,
                gate1_ok=bool(upgrades),
                mean_before=sb["mean"], mean_after=sa["mean"],
                gate2_ceiling=1.10 * sb["mean"], gate2_mean_ok=mean_ok,
                mean_move_in_seed_sd=round(abs(d_mean) / MEAN_SEED_SD, 2),
                rce_protected_ok=rce_ok,
                rce_ungateable_on_one_run=True,
                verdict="SUCCESS" if (upgrades and mean_ok and rce_ok) else "FAIL",
            )
            md += ["verdict: " + ", ".join(f"{k}={v}" for k, v in verdict.items()), ""]
    pd.DataFrame(wide).reindex(METRIC_ORDER).to_csv(os.path.join(OUT_DIR, "before_after.csv"))
    with open(os.path.join(OUT_DIR, "before_after.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
