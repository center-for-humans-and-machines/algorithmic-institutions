"""Before/after table for the contribution copula recalibration
(auto/contribution-copula-recalibrated).

Before = the parent's `_simtimeout` frontier sim (the serving-path fix), after
= the `_copularecal` sim (the same stack with a refitted `copula_rho` stamped
on a copy of the contributor). Nothing is retrained in either: the two runs
differ in three scalar fields of one artifact and in nothing else, so the
run-to-run variation under a fixed seed is zero, not the 0.138 of a retrain.
The PR #195 seed floor is still the right scale for READING a row, because it
is the scale on which the model is uncertain.

The protected row is judged under the AMENDED rule (notes/autoresearch.md §2
on `docs/post-rebaseline-program`): no movement smaller than its row's seed
deviation counts either for or against an experiment, all three protected-row
clauses take that threshold, and the sign clause is retired on the 10-14 and
15-19 bands. The raw (pre-amendment) verdicts stay visible beside them.

Usage (repo root, PYTHONPATH=src):
    python scripts/data_analysis/copula_recal_table.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from curpun_rebaseline import (  # noqa: E402
    METRIC_ORDER,
    SKIP,
    band,
    human_rce_fit,
    md_table,
    read_scores,
    summarise,
)
from punisher_timeout_table import (  # noqa: E402
    MEAN_SEED_SD,
    RCE_SLOPE_SEED_SD,
    ROWS_LE1_SEED_SD,
    SEED_SD,
    UNGATEABLE,
    rce_check,
)

OUT_DIR = "plots/data_analysis/evaluation/contribution_copula_recalibrated"
BEFORE, AFTER = "_simtimeout", "_copularecal"
RUN = "lin_multinomial_copula_self"
TARGETS = ["CG"]  # declared in the log before the refit
WATCH = ["SC", "RCC"]
# The sign clause is retired on these bands: retraining an unchanged model
# flips them on its own (PR #195, six arms).
SIGN_BANDS = ["0-4", "5-9"]


def amend(checks, d_score, d_slope):
    """The amended protected-row rule on top of `rce_check`'s raw verdicts.

    Every clause takes its row's seed deviation as a threshold: the band-drop
    clause RCE's own 0.106, the two slope clauses their band's slope sd. The
    sign clause additionally applies only on the 0-4 and 5-9 bands."""
    out = dict(checks)
    out["band_downgrade_raw"] = checks["band_downgrade"]
    out["band_downgrade"] = bool(
        checks["band_downgrade"] and abs(d_score) > SEED_SD["RCE"]
    )
    for key in ("sign_lost", "magnitude_eroded"):
        out[key + "_raw"] = checks[key]
        out[key] = [
            k
            for k in checks[key]
            if abs(d_slope[k]) > RCE_SLOPE_SEED_SD[k]
            and (key != "sign_lost" or k in SIGN_BANDS)
        ]
    out["fires"] = bool(
        out["band_downgrade"] or out["sign_lost"] or out["magnitude_eroded"]
    )
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    human_fit = human_rce_fit()
    before = read_scores(f"plots/simulation/{SKIP}{BEFORE}/evaluation/scores.csv", RUN)
    after = read_scores(f"plots/simulation/{SKIP}{AFTER}/evaluation/scores.csv", RUN)
    assert before is not None and after is not None, "missing scores.csv"

    t = pd.DataFrame({"before": before, "after": after})
    t["delta"] = t["after"] - t["before"]
    t["band"] = [
        f"{band(b)} -> {band(a)}" if band(b) != band(a) else band(b)
        for b, a in zip(t.before, t.after)
    ]
    t["seed_sd"] = [SEED_SD.get(r, np.nan) for r in t.index]
    t["in_seed_sd"] = (t["delta"].abs() / t["seed_sd"]).round(2)
    t["legible"] = t["in_seed_sd"] >= 1.0
    t["ungateable"] = [r in UNGATEABLE for r in t.index]
    t["target"] = [r in TARGETS for r in t.index]
    sb, sa = summarise(before), summarise(after)
    d_mean = sa["mean"] - sb["mean"]
    d_rows = sa["rows <= 1"] - sb["rows <= 1"]
    t.loc["mean"] = [sb["mean"], sa["mean"], d_mean, "", MEAN_SEED_SD,
                     round(abs(d_mean) / MEAN_SEED_SD, 2),
                     abs(d_mean) >= MEAN_SEED_SD, False, False]  # fmt: skip
    t.loc["rows <= 1"] = [sb["rows <= 1"], sa["rows <= 1"], d_rows, "",
                          ROWS_LE1_SEED_SD, round(abs(d_rows) / ROWS_LE1_SEED_SD, 2),
                          abs(d_rows) >= ROWS_LE1_SEED_SD, True, False]  # fmt: skip
    t.index.name = "row"

    rce_rows, checks = rce_check(
        human_fit, SKIP + BEFORE, SKIP + AFTER, RUN, before["RCE"], after["RCE"]
    )
    rce = pd.DataFrame(rce_rows).set_index("stage")
    slopes = {s: r for s, r in zip(rce.index, rce_rows)}
    d_slope = {
        k: float(slopes["after"][k].split()[0]) - float(slopes["before"][k].split()[0])
        for k in RCE_SLOPE_SEED_SD
    }
    checks = amend(checks, after["RCE"] - before["RCE"], d_slope)
    rce.index.name = "RCE slopes"

    upgrades = [
        r for r in TARGETS
        if band(after[r]) != band(before[r]) and after[r] < before[r]
    ]  # fmt: skip
    cleared = [r for r in upgrades if abs(after[r] - before[r]) > SEED_SD[r]]
    mean_ok = sa["mean"] <= 1.10 * sb["mean"]
    verdict = dict(
        targets=TARGETS,
        target_before={r: round(before[r], 4) for r in TARGETS},
        target_after={r: round(after[r], 4) for r in TARGETS},
        target_in_seed_sd={
            r: round(abs(after[r] - before[r]) / SEED_SD[r], 2) for r in TARGETS
        },
        gate1_band_upgrades=upgrades,
        gate1_upgrades_clearing_the_floor=cleared,
        gate1_ok=bool(cleared),
        watch={r: (round(before[r], 4), round(after[r], 4)) for r in WATCH},
        mean_before=round(sb["mean"], 4), mean_after=round(sa["mean"], 4),
        gate2_ceiling=round(1.10 * sb["mean"], 4), gate2_mean_ok=bool(mean_ok),
        mean_move_in_seed_sd=round(abs(d_mean) / MEAN_SEED_SD, 2),
        rce_protected_ok=not checks["fires"],
        rce_ungateable_on_one_run=True,
        verdict="SUCCESS" if (cleared and mean_ok and not checks["fires"]) else "FAIL",
    )  # fmt: skip

    md = [
        "### frontier: PR 181 stimulus skip x joint-exodus GNN switch, "
        "lin_multinomial copula",
        "",
        f"before = `{SKIP}{BEFORE}`, after = `{SKIP}{AFTER}`, run `{RUN}`",
        "",
        md_table(t, "{:.4f}"),
        "",
        md_table(rce, "{:+.3f}"),
        "",
        "protected-row checks (amended rule): "
        + ", ".join(f"{k}={v}" for k, v in checks.items()),
        "",
        "verdict: " + ", ".join(f"{k}={v}" for k, v in verdict.items()),
        "",
    ]
    pd.DataFrame(
        {"before": before, "after": after}
    ).reindex(METRIC_ORDER).to_csv(os.path.join(OUT_DIR, "before_after.csv"))
    with open(os.path.join(OUT_DIR, "before_after.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
