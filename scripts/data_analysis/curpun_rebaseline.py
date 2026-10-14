"""Before/after table for the punisher-current-contribution re-baseline.

Every AH punisher so far conditioned on the previous round's contribution;
the retrained punishers condition on the current one. For each rerun case
(a-e, notes/autoresearch_log/punisher-current-contribution-cases.md) this
reads the old-punisher ("before") scores -- the source sims rescored with the
22-row suite (RCE included), which stage B wrote to
plots/data_analysis/evaluation/punisher_current_contr/before/<case>/scores.csv
-- and the retrained-punisher ("after") scores from the `_curpun` sim dir's
evaluation/scores.csv (python -m aimanager evaluate <config>), and writes to
plots/data_analysis/evaluation/punisher_current_contr/:

- rebaseline_table.csv: 22 rows x (case, before / after / delta), plus the
  22-row mean and the rows <= 1 count
- rce_bands.csv: per-band RCE slopes (0-4 / 5-9 / 10-14 / 15-19) of every
  before and after sim next to the human ones, with the sign pattern
- rebaseline_table.md: the same as markdown (one table per case, a summary
  table, and the RCE band table)

Cases whose after sim is missing are reported with the before column only.

Usage (repo root):
    python scripts/data_analysis/curpun_rebaseline.py [--suite-src <src dir>]

--suite-src prepends a source tree carrying the RCE row (for running
before the suite on this branch has it).
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

OUT_DIR = "plots/data_analysis/evaluation/punisher_current_contr"
BEFORE_DIR = os.path.join(OUT_DIR, "before")
SIM = "plots/simulation"
RUN = "ah group_switching managed by "

METRIC_ORDER = [
    "CA", "CB", "CC", "CD", "CE", "CF", "CG",
    "SA", "SB", "SC",
    "PA", "PB", "PC", "PD",
    "RCA", "RCB", "RCC", "RCD", "RCE",
    "RSA", "RPA", "RPB",
]  # fmt: skip
BAND_EDGES = [0, 1, 2, 5, np.inf]
BAND_LABELS = ["<= 1", "1-2", "2-5", "> 5"]

VNODE = "23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch"
SKIP = "23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch"
INFL = (
    "23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus"
    "_k_onehot_switch"
)
KEXO = (
    "23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus"
    "_k_onehot_switch"
)
MAIN = "23_2g8a_self_gnn_contr_gnn_switch"

# id, label, before-scores case dir, before sim dir, after sim dir, run name
CASES = [
    ("a_vnode", "a: PR 179 group vnode, lin_multinomial copula", "a_vnode",
     VNODE, VNODE + "_curpun", "lin_multinomial_copula_self"),
    ("b_skip", "b: PR 181 stimulus skip, lin_multinomial copula", "b_skip",
     SKIP, SKIP + "_curpun", "lin_multinomial_copula_self"),
    ("c_infl", "c: PR 177 inflated gmlp, lin_multinomial copula", "c_infl",
     INFL, INFL + "_curpun", "lin_multinomial_copula_self"),
    ("d_kexo", "d: PR 174 k-one-hot switch on gmlp copula, lin_multinomial copula",
     "d_kexo", KEXO, KEXO + "_curpun", "lin_multinomial_copula_self"),
    ("e_lin", "e: main gnn x gnn, lin_multinomial (no copula)", "e_main",
     MAIN, MAIN + "_curpun", "lin_multinomial_self"),
    ("e_gnn", "e: main gnn x gnn, gnn punisher", "e_main",
     MAIN, MAIN + "_curpun", "gnn_self"),
]  # fmt: skip


def band(score):
    if pd.isna(score):
        return "nan"
    return BAND_LABELS[int(np.digitize(score, BAND_EDGES[1:-1]))]


def read_scores(path, run):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["run"] == RUN + run].set_index("metric")["score"]
    missing = [m for m in METRIC_ORDER if m not in df.index]
    if missing:
        warnings.warn(f"{path}: rows missing for {run}: {missing}")
    return df.reindex(METRIC_ORDER)


def summarise(s):
    return pd.Series(
        {
            "mean": s.mean(),
            "rows <= 1": float((s <= 1).sum()) if s.notna().any() else np.nan,
        },
    )


def rce_bands(human_fit, before_dir, after_dir, run):
    """Per-band RCE slopes for the before and after sim of one case."""
    from aimanager.evaluation_suite.convert import load_sim
    from aimanager.evaluation_suite.metrics import GROUPS

    rows = []
    human_sign = np.sign(human_fit["slope"])
    for stage, sim_dir in [("before", before_dir), ("after", after_dir)]:
        parquet = os.path.join(SIM, sim_dir, "per_round.parquet")
        if not os.path.exists(parquet):
            continue
        fit = GROUPS["R"]._rce_fit(load_sim(parquet)[RUN + run])
        signs = "".join("+" if s > 0 else "-" for s in fit["slope"])
        match = int((np.sign(fit["slope"]) == human_sign).sum())
        rows.append(
            {
                "stage": stage,
                **{f"slope_{b}": v for b, v in fit["slope"].items()},
                **{f"n_{b}": v for b, v in fit["n"].items()},
                "signs": signs,
                "signs_vs_human": f"{signs} ({match}/4)",
            }
        )
    return rows


def human_rce_fit():
    from aimanager.evaluation_suite.convert import HUMAN_DATA_FILE, load_human
    from aimanager.evaluation_suite.metrics import GROUPS

    return GROUPS["R"]._rce_fit(load_human(HUMAN_DATA_FILE))


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for idx, row in df.iterrows():
        cells = [
            floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-src", help="source tree with the RCE row")
    args = parser.parse_args()
    if args.suite_src:
        sys.path.insert(0, args.suite_src)
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        human_fit = human_rce_fit()
    except (ImportError, AttributeError) as e:  # suite without RCE
        warnings.warn(f"RCE band slopes skipped: {e}")
        human_fit = None

    wide, summary, bands, md = {}, [], [], []
    for case, label, before_case, before_dir, after_dir, run in CASES:
        before = read_scores(os.path.join(BEFORE_DIR, before_case, "scores.csv"), run)
        after = read_scores(
            os.path.join(SIM, after_dir, "evaluation", "scores.csv"), run
        )
        if before is None:
            warnings.warn(f"{case}: before scores missing, skipped")
            continue
        table = pd.DataFrame({"before": before})
        table["after"] = after if after is not None else np.nan
        table["delta"] = table["after"] - table["before"]
        table["band"] = [
            f"{band(b)} -> {band(a)}" if not pd.isna(a) else band(b)
            for b, a in zip(table["before"], table["after"])
        ]
        table.index.name = "metric"
        wide.update({f"{case}_{c}": table[c] for c in ["before", "after", "delta"]})

        s_before, s_after = summarise(table["before"]), summarise(table["after"])
        row = {
            "case": case,
            "label": label,
            "after_available": after is not None,
            "mean_before": s_before["mean"],
            "mean_after": s_after["mean"],
            "mean_delta": s_after["mean"] - s_before["mean"],
            "rows_le1_before": int(s_before["rows <= 1"]),
            "rows_le1_after": int(s_after["rows <= 1"]) if after is not None else None,
            "RCE_before": table.loc["RCE", "before"],
            "RCE_after": table.loc["RCE", "after"],
            "RCE_band": table.loc["RCE", "band"],
        }
        summary.append(row)

        if human_fit is not None:
            for b in rce_bands(human_fit, before_dir, after_dir, run):
                bands.append({"case": case, **b})

        md.append(f"### {label}\n")
        md.append(
            f"Run `{RUN + run}`; before `{SIM}/{before_dir}` (rescored, "
            f"`{BEFORE_DIR}/{before_case}/scores.csv`), after `{SIM}/{after_dir}`.\n"
        )
        full = pd.concat([table, pd.DataFrame({"before": s_before, "after": s_after})])
        full.loc["mean", "delta"] = (
            full.loc["mean", "after"] - full.loc["mean", "before"]
        )
        full.loc["rows <= 1", "delta"] = (
            full.loc["rows <= 1", "after"] - full.loc["rows <= 1", "before"]
        )
        full["band"] = full["band"].fillna("")
        full.index.name = "metric"
        md.append(md_table(full) + "\n")

    wide_df = pd.DataFrame(wide)
    wide_df.index.name = "metric"
    extra = pd.DataFrame(
        {c: summarise(wide_df[c]) for c in wide_df.columns if not c.endswith("_delta")}
    )
    for c in [c for c in wide_df.columns if c.endswith("_delta")]:
        extra[c] = (
            extra[c[: -len("_delta")] + "_after"]
            - extra[c[: -len("_delta")] + "_before"]
        )
    wide_df = pd.concat([wide_df, extra[wide_df.columns]])
    wide_df.to_csv(os.path.join(OUT_DIR, "rebaseline_table.csv"))

    summary_df = pd.DataFrame(summary).set_index("case")
    header = [
        "# Punisher current-contribution re-baseline\n",
        "Scores are multiples of the human noise ceiling (<= 1 at the ceiling, 1-2 "
        "minor, 2-5 clear, > 5 not reproduced); delta = after - before, negative is "
        "an improvement. `before` = the source sim with the prev-contribution "
        "punisher, rescored with the 22-row suite; `after` = the same "
        "contributor/switch stack with the punisher retrained on the current "
        "contribution.\n",
        "## Summary\n",
        md_table(summary_df.drop(columns=["label"])) + "\n",
    ]
    if bands:
        bands_df = pd.DataFrame(bands)
        bands_df.to_csv(os.path.join(OUT_DIR, "rce_bands.csv"), index=False)
        human_row = pd.DataFrame(
            [
                {
                    "case": "human",
                    "stage": "",
                    **{f"slope_{b}": v for b, v in human_fit["slope"].items()},
                    "signs": "".join("+" if s > 0 else "-" for s in human_fit["slope"]),
                }
            ]
        )
        show = pd.concat([human_row, bands_df]).set_index("case")
        show = show[
            [c for c in show.columns if c.startswith(("stage", "slope_", "signs"))]
        ]
        header += [
            "## RCE per band: OLS slope of next-round change on punishment received\n",
            "Human pattern: comply when punished at low contribution, withdraw at "
            "high (++--). `signs_vs_human` counts matching signs.\n",
            md_table(show.fillna("")) + "\n",
        ]
    header.append("## Per case\n")
    with open(os.path.join(OUT_DIR, "rebaseline_table.md"), "w") as f:
        f.write("\n".join(header + md))
    print(summary_df.drop(columns=["label"]).to_string())
    print(
        f"-> {OUT_DIR}/rebaseline_table.{{csv,md}}"
        + (", rce_bands.csv" if bands else "")
    )


if __name__ == "__main__":
    main()
