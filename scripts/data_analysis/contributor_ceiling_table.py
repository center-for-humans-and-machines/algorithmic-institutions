"""Before/after tables for the contributor ceiling indicator.

Before = the parent branch's frontier `_ceiling` sim (PR #192's punisher with
the stimulus-skip contributor as it stands), after = the same stack with the
contributor retrained on `prev_contribution_max`. Writes to
plots/data_analysis/evaluation/contributor_ceiling_indicator/:

- `before_after.csv` / `.md`: 22 rows x (before / after / delta / band), the
  mean and rows <= 1, the RCE band slopes with their standard errors and row
  counts, the protected-row checks and the gate verdict;
- `ceiling_decomposition.csv`: RCC split into its two means and their
  populations -- the table PR #192's section 4 leaves as this experiment's
  baseline, reproduced line for line for the human row, the baseline and the
  candidate.

The magnitude clause of the protected-row rule is the AMENDED one: a slope
whose magnitude halves only counts as erosion when it did NOT move closer to
the human value AND the change exceeds one pooled standard error. The clause
misfired twice on PR #192's reference stacks -- once on a slope crossing zero
toward the human value, once at 0.90 pooled SE -- and its section 4 asks for
exactly this amendment.

Usage (repo root, PYTHONPATH=src):
    python scripts/data_analysis/contributor_ceiling_table.py
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
    human_rce_fit,
    md_table,
    read_scores,
    summarise,
)

OUT_DIR = "plots/data_analysis/evaluation/contributor_ceiling_indicator"
BEFORE = "23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling"
AFTER = "23_2g8a_contr_ceilind_self_gnncopar1_contr_gnn_switch_ceiling"
RUN_NAME = "lin_multinomial_copula_self"
TARGET = "RCC"


# --------------------------------------------------------------------------- #
# RCC, decomposed into the two means and their populations
# --------------------------------------------------------------------------- #
def ceiling_decomposition(df):
    from aimanager.evaluation_suite.metrics import GROUPS

    d = GROUPS["R"]._with_dc(df)
    full = d[(d["contribution"] == 20) & d["dc"].notna() & d["punishment"].notna()]
    p, u = full[full["punishment"] > 0], full[full["punishment"] == 0]
    return {
        "contrast": p["dc"].mean() - u["dc"].mean(),
        "dc_punished": p["dc"].mean(),
        "n_punished": len(p),
        "dc_unpunished": u["dc"].mean(),
        "n_unpunished": len(u),
        "punished_share": len(p) / len(full),
    }


def decomposition_table():
    from aimanager.evaluation_suite.convert import (
        HUMAN_DATA_FILE,
        load_human,
        load_sim,
    )

    rows = [{"case": "human", **ceiling_decomposition(load_human(HUMAN_DATA_FILE))}]
    for case, d in (("baseline (PR 192)", BEFORE), ("candidate", AFTER)):
        p = os.path.join(SIM, d, "per_round.parquet")
        if not os.path.exists(p):
            print(f"{case}: no per_round.parquet at {p}")
            continue
        rows.append(
            {"case": case, **ceiling_decomposition(load_sim(p)[RUN + RUN_NAME])}
        )
    return pd.DataFrame(rows).set_index("case")


# --------------------------------------------------------------------------- #
# RCE: the protected row, with the amended magnitude clause
# --------------------------------------------------------------------------- #
def _slope_row(stage, fit):
    return {
        "stage": stage,
        **{
            k: f"{fit.loc[k, 'slope']:+.3f} +- {fit.loc[k, 'se']:.3f} "
            f"(n {int(fit.loc[k, 'n'])})"
            for k in fit.index
        },
    }


def rce_check(human_fit, before_score, after_score):
    from aimanager.evaluation_suite.convert import load_sim
    from aimanager.evaluation_suite.metrics import GROUPS

    full = {}
    for stage, d in (("before", BEFORE), ("after", AFTER)):
        p = os.path.join(SIM, d, "per_round.parquet")
        if os.path.exists(p):
            full[stage] = GROUPS["R"]._rce_fit(load_sim(p)[RUN + RUN_NAME])
    rows = [_slope_row("human", human_fit)]
    for stage, f in full.items():
        rows.append(_slope_row(stage, f))
    if len(full) < 2:
        return rows, {}
    b, a = full["before"]["slope"], full["after"]["slope"]
    se_b, se_a = full["before"]["se"], full["after"]["se"]
    h, hs = human_fit["slope"], np.sign(human_fit["slope"])
    change_in_se = {
        k: float(abs(a[k] - b[k]) / np.hypot(se_b[k], se_a[k])) for k in a.index
    }
    closer = {k: abs(a[k] - h[k]) < abs(b[k] - h[k]) for k in a.index}
    halved = [k for k in a.index if abs(a[k]) <= 0.5 * abs(b[k])]
    return rows, {
        "band_downgrade": band(after_score) != band(before_score)
        and after_score > before_score,
        "sign_lost": [
            k for k in a.index if np.sign(a[k]) != hs[k] and np.sign(b[k]) == hs[k]
        ],
        # the amended clause: halving alone is not erosion
        "magnitude_halved_raw": halved,
        "magnitude_eroded": [
            k for k in halved if not closer[k] and change_in_se[k] > 1.0
        ],
        "signs_before": "".join("+" if v > 0 else "-" for v in b),
        "signs_after": "".join("+" if v > 0 else "-" for v in a),
        "change_in_se": {k: round(v, 2) for k, v in change_in_se.items()},
        "closer_to_human": closer,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    md = []

    before = read_scores(os.path.join(SIM, BEFORE, "evaluation/scores.csv"), RUN_NAME)
    after = read_scores(os.path.join(SIM, AFTER, "evaluation/scores.csv"), RUN_NAME)
    assert before is not None, f"no baseline scores under {BEFORE}"
    if after is None:
        print(f"candidate scores missing under {AFTER} -- run evaluate first")
        return

    t = pd.DataFrame({"before": before, "after": after})
    t["delta"] = t["after"] - t["before"]
    t["band"] = [
        f"{band(b)} -> {band(a)}" if band(b) != band(a) else band(b)
        for b, a in zip(t.before, t.after)
    ]
    sb, sa = summarise(before), summarise(after)
    t.loc["mean"] = [sb["mean"], sa["mean"], sa["mean"] - sb["mean"], ""]
    t.loc["rows <= 1"] = [
        sb["rows <= 1"],
        sa["rows <= 1"],
        sa["rows <= 1"] - sb["rows <= 1"],
        "",
    ]
    t.index.name = "row"
    pd.DataFrame({"before": before, "after": after}).reindex(METRIC_ORDER).to_csv(
        os.path.join(OUT_DIR, "before_after.csv")
    )
    md += [
        "### The 22 rows, frontier stack (the gated one)",
        "",
        md_table(t, "{:.4f}"),
        "",
    ]

    dec = decomposition_table()
    dec.to_csv(os.path.join(OUT_DIR, "ceiling_decomposition.csv"))
    md += ["### RCC decomposed at the ceiling", "", md_table(dec, "{:.4f}"), ""]

    human_fit = human_rce_fit()
    rce_rows, checks = rce_check(human_fit, before["RCE"], after["RCE"])
    rce = pd.DataFrame(rce_rows).set_index("stage")
    rce.index.name = "RCE slopes"
    md += ["### RCE band slopes (protected)", "", md_table(rce), ""]
    md += [f"protected-row checks: {checks}", ""]

    mean_ok = sa["mean"] <= 1.10 * sb["mean"]
    up = band(after[TARGET]) != band(before[TARGET]) and after[TARGET] < before[TARGET]
    rce_ok = not (
        checks["band_downgrade"] or checks["sign_lost"] or checks["magnitude_eroded"]
    )
    verdict = dict(
        target=TARGET,
        target_before=before[TARGET],
        target_after=after[TARGET],
        gate1_band_upgrade=up,
        mean_before=sb["mean"],
        mean_after=sa["mean"],
        gate2_ceiling=1.10 * sb["mean"],
        gate2_mean_ok=mean_ok,
        rce_protected_ok=rce_ok,
        verdict="SUCCESS" if (up and mean_ok and rce_ok) else "FAIL",
    )
    md += ["verdict: " + ", ".join(f"{k}={v}" for k, v in verdict.items()), ""]

    with open(os.path.join(OUT_DIR, "before_after.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
