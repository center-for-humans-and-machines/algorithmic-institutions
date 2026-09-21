"""Before/after table for the free-punishment fix (auto/free-punishment-fix).

Before = the parent chain's `_simtimeout` sims (PR #205 / #196), after = the
`_freepun` sims. The two differ in *nothing* but `environment.punish` and the
one line of `simulate.py` that records its result: same artifacts, same seed,
same episode count, nothing retrained and no copula recalibrated. The
run-to-run variation of that pair is therefore exactly zero -- the simulation
is bit-reproducible, which the campaign established -- so every movement below
is the fix and nothing else. The PR #195 seed floor is still quoted beside it,
because it is the scale on which the *model* is uncertain: a row that moves
less than one seed sd is not distinguishable from an unchanged model retrained
with another draw, and ten rows cannot be gated on a single run at all.

Unlike its two predecessors this fix reaches the recorded output. The script
therefore also diffs the two `per_round.parquet` frames cell by cell and
reports how much of the movement is the direct correction (punishment cells
the game never charged) and how much is the simulation diverging afterwards.

Writes to plots/data_analysis/evaluation/free_punishment/:

- before_after.csv / .md: 22 rows x (before / after / delta / band / seed_sd /
  in_seed_sd / legible / ungateable / target), the mean and rows <= 1, the
  parquet diff, and the RCE band slopes with the protected-row checks;
- the gate verdict for the frontier case.

The protected-row clauses are the **amended** ones (notes/autoresearch.md §2
on docs/post-rebaseline-program): the band-drop clause fires only on a drop
larger than RCE's own seed sd (0.106); the sign clause is retired on the
10-14 and 15-19 bands, where retraining an unchanged model flips the sign on
its own, and kept on 0-4 and 5-9; the magnitude clause fires only when the
candidate's slope is not closer to the human value and the change exceeds one
pooled standard error. Every band is reported with its slope, its standard
error and its row count so a firing can be read.

Usage (repo root, PYTHONPATH=src):
    python scripts/data_analysis/free_punishment_table.py
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
from punisher_timeout_table import (  # noqa: E402
    MEAN_SEED_SD,
    RCE_SLOPE_SEED_SD,
    ROWS_LE1_SEED_SD,
    SEED_SD,
    UNGATEABLE,
    rce_check,
)

OUT_DIR = "plots/data_analysis/evaluation/free_punishment"
BEFORE, AFTER = "_simtimeout", "_freepun"
# Declared in the log before the first simulation: the group-spread row and
# the response family. The fix removes a punishment the game never charged
# from the contribution model's only channel from the manager, so if it shows
# anywhere it shows in how the contributor responds to punishment.
TARGETS = ["CG", "RCA", "RCB", "RCC", "RCD", "RCE", "RSA", "RPA", "RPB"]
# The sign clause survives only where the six seed arms of PR #195 agree.
RCE_SIGN_BANDS = ["0-4", "5-9"]
CASES = [
    ("frontier", "PR 181 stimulus skip x joint-exodus GNN switch, lin_multinomial copula",
     SKIP, "lin_multinomial_copula_self", True),
    ("ref_lin", "main gnn x gnn, lin_multinomial (timeout feature)", MAIN,
     "lin_multinomial_self", False),
    ("ref_gnn", "main gnn x gnn, gnn punisher (timeout feature)", MAIN,
     "gnn_self", False),
]  # fmt: skip


def parquet_diff(src, run):
    """How much of the recorded output actually moved, and where."""
    from aimanager.evaluation_suite.convert import load_sim

    paths = [os.path.join(SIM, src + s, "per_round.parquet") for s in (BEFORE, AFTER)]
    if not all(os.path.exists(p) for p in paths):
        return {}
    b, a = (load_sim(p)[RUN + run] for p in paths)
    key = ["episode_id", "participant_code", "round_number"]
    b, a = b.sort_values(key).reset_index(drop=True), a.sort_values(key).reset_index(drop=True)
    out = {"rows": len(b)}
    for col in ("punishment", "contribution", "group_id"):
        d = b[col].fillna(-1) != a[col].fillna(-1)
        out[f"{col}_changed"] = int(d.sum())
        out[f"{col}_changed_share"] = round(float(d.mean()), 6)
    out["mean_punishment_before"] = round(float(b["punishment"].mean()), 4)
    out["mean_punishment_after"] = round(float(a["punishment"].mean()), 4)
    out["mean_contribution_before"] = round(float(b["contribution"].mean()), 4)
    out["mean_contribution_after"] = round(float(a["contribution"].mean()), 4)
    # the direct correction: punishment cells that went to 0 and nothing else
    direct = (b["punishment"] > 0) & (a["punishment"] == 0)
    out["punishment_cells_zeroed"] = int(direct.sum())
    return out


def amended_rce(checks, before_score, after_score):
    """The amended protected-row clauses (notes/autoresearch.md §2)."""
    if not checks:
        return {}
    drop = after_score - before_score
    return {
        "band_downgrade_raw": checks["band_downgrade"],
        "band_downgrade_amended": bool(
            checks["band_downgrade"] and drop > SEED_SD["RCE"]
        ),
        "band_drop_in_seed_sd": round(abs(drop) / SEED_SD["RCE"], 2),
        "sign_lost_raw": checks["sign_lost"],
        "sign_lost_amended": [b for b in checks["sign_lost"] if b in RCE_SIGN_BANDS],
        "magnitude_halved_raw": checks["magnitude_halved"],
        "magnitude_eroded_amended": checks["magnitude_eroded"],
        "magnitude_halved_but_toward_human": checks["magnitude_halved_but_toward_human"],
        "change_in_se": checks["change_in_se"],
        "change_in_seed_sd": checks["change_in_seed_sd"],
        "rce_ungateable_on_one_run": True,
        "rce_seed_sd": SEED_SD["RCE"],
        "band_slope_seed_sd": RCE_SLOPE_SEED_SD,
    }


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
        amended = amended_rce(checks, before["RCE"], after["RCE"])
        diff = parquet_diff(src, run)
        md += [f"### {case}: {label}", "", md_table(t, "{:.4f}"), ""]
        md += [f"recorded output diff: {diff}", ""]
        rce = pd.DataFrame(rce_rows).set_index("stage")
        rce.index.name = "RCE slopes"
        md += [md_table(rce, "{:+.3f}"), ""]
        md += [f"protected-row checks (raw): {checks}", ""]
        md += [f"protected-row checks (amended): {amended}", ""]
        if gated:
            mean_ok = sa["mean"] <= 1.10 * sb["mean"]
            # a band upgrade only counts when it also clears the row's floor
            upgrades = [
                r for r in TARGETS
                if band(after[r]) != band(before[r])
                and after[r] < before[r]
                and abs(after[r] - before[r]) >= SEED_SD[r]
            ]
            rce_ok = not (
                amended["band_downgrade_amended"]
                or amended["sign_lost_amended"]
                or amended["magnitude_eroded_amended"]
            )
            verdict = dict(
                targets=TARGETS,
                target_before={r: round(before[r], 4) for r in TARGETS},
                target_after={r: round(after[r], 4) for r in TARGETS},
                target_in_seed_sd={
                    r: round(abs(after[r] - before[r]) / SEED_SD[r], 2) for r in TARGETS
                },
                target_legible=[
                    r for r in TARGETS if abs(after[r] - before[r]) >= SEED_SD[r]
                ],
                gate1_band_upgrades=upgrades,
                gate1_ok=bool(upgrades),
                mean_before=sb["mean"], mean_after=sa["mean"],
                gate2_ceiling=1.10 * sb["mean"], gate2_mean_ok=mean_ok,
                mean_move_in_seed_sd=round(abs(d_mean) / MEAN_SEED_SD, 2),
                rce_protected_ok=rce_ok,
                verdict="SUCCESS" if (upgrades and mean_ok and rce_ok) else "FAIL",
            )
            md += ["verdict: " + ", ".join(f"{k}={v}" for k, v in verdict.items()), ""]
    pd.DataFrame(wide).reindex(METRIC_ORDER).to_csv(os.path.join(OUT_DIR, "before_after.csv"))
    with open(os.path.join(OUT_DIR, "before_after.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
