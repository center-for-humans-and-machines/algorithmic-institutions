"""Row-by-row comparison of the switch-kexo-port candidate against the frontier
baseline (PR #181's stack as re-baselined by auto/punisher-current-contribution),
in the format of plots/data_analysis/evaluation/punisher_current_contr/
rebaseline_table.md. Reuses the helpers of curpun_rebaseline.py. Writes
plots/data_analysis/evaluation/switch_kexo_port/{compare_table.md,compare.csv,
rce_bands.csv}. Run with PYTHONPATH=src from the branch's worktree (the RCE
row and its band fit live in this tree's evaluation suite).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from curpun_rebaseline import (  # noqa: E402
    SIM,
    band,
    human_rce_fit,
    md_table,
    rce_bands,
    read_scores,
    summarise,
)

OUT_DIR = "plots/data_analysis/evaluation/switch_kexo_port"
RUN = "lin_multinomial_copula_self"
BEFORE = "23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun"
AFTER = (
    "23_2g8a_switch_kexo_port_self_gnncopar1_contr_stimulus_skip_contr_gnn_kexo"
    "_switch_curpun"
)
GATE_MARGIN = 1.10
TARGETS = ["SC", "RCD"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    before = read_scores(os.path.join(SIM, BEFORE, "evaluation", "scores.csv"), RUN)
    after = read_scores(os.path.join(SIM, AFTER, "evaluation", "scores.csv"), RUN)
    table = pd.DataFrame({"before": before, "after": after})
    table["delta"] = table["after"] - table["before"]
    table["band"] = [
        f"{band(b)} -> {band(a)}" for b, a in zip(table["before"], table["after"])
    ]
    sb, sa = summarise(before), summarise(after)
    for k in ["mean", "rows <= 1"]:
        table.loc[k] = [sb[k], sa[k], sa[k] - sb[k], ""]
    table.index.name = "metric"
    table.to_csv(os.path.join(OUT_DIR, "compare.csv"))

    human = human_rce_fit()
    bands = pd.DataFrame(rce_bands(human, BEFORE, AFTER, RUN)).set_index("stage")
    bands.loc["human"] = {
        **{f"slope_{b}": v for b, v in human["slope"].items()},
        **{f"n_{b}": v for b, v in human["n"].items()},
        "signs": "".join("+" if s > 0 else "-" for s in human["slope"]),
        "signs_vs_human": "",
    }
    bands = bands.loc[["human", "before", "after"]]
    bands.to_csv(os.path.join(OUT_DIR, "rce_bands.csv"))

    # Protected-row check on RCE: band, sign per band, magnitude halving.
    slope_cols = [c for c in bands.columns if c.startswith("slope_")]
    hs = np.sign(human["slope"].values)
    sb_ = bands.loc["before", slope_cols].astype(float).values
    sa_ = bands.loc["after", slope_cols].astype(float).values
    rce_fail = {
        "band drop": band(after["RCE"]) != band(before["RCE"])
        and after["RCE"] > before["RCE"],
        "sign lost": bool(((np.sign(sa_) != hs) & (np.sign(sb_) == hs)).any()),
        "magnitude halved": bool((np.abs(sa_) <= 0.5 * np.abs(sb_)).any()),
    }
    gate1 = {
        r: band(after[r]) != band(before[r]) and after[r] < before[r] for r in TARGETS
    }
    gate2 = sa["mean"] <= GATE_MARGIN * sb["mean"]
    ok = any(gate1.values()) and gate2 and not any(rce_fail.values())
    verdict = "SUCCESS" if ok else "FAIL"

    md = [
        "# switch-kexo-port: k-one-hot joint-exodus switch in the frontier stack\n",
        "Scores are multiples of the human noise ceiling (<= 1 at the ceiling, "
        "1-2 minor, 2-5 clear, > 5 not reproduced); delta = after - before, "
        "negative is an improvement. `before` = the frontier baseline "
        f"`{BEFORE}`; `after` = the same stack with only the switch model "
        f"swapped to `switch_exodus_k_onehot`, `{AFTER}`. Run `{RUN}`.\n",
        "## Per row\n",
        md_table(table),
        "\n## RCE per band: OLS slope of next-round change on punishment received\n",
        md_table(bands[slope_cols + ["signs", "signs_vs_human"]]),
        "\n## Gates\n",
        f"- gate 1 (band upgrade on {' or '.join(TARGETS)}): {gate1}",
        f"- gate 2 (mean <= {GATE_MARGIN:.2f} x {sb['mean']:.4f} = "
        f"{GATE_MARGIN * sb['mean']:.4f}): after mean {sa['mean']:.4f} -> {gate2}",
        f"- RCE protected-row failures: {rce_fail}",
        f"- verdict: **[{verdict}]**",
    ]
    with open(os.path.join(OUT_DIR, "compare_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
