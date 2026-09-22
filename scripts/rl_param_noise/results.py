"""Read the finished runs and answer the arm's question.

Three outputs, in the order they decide anything:

  results_shape.csv   DID THE SHAPE RECOVER. Mean punishment per contribution
                      bin for each seed's EVALUATED policy -- the fully
                      deterministic rollout, every exploration mechanism off --
                      pooled over a late window of evaluation points, with the
                      three-statistic check from targeting.py beside it. The
                      artificial punisher measured on the opponent's group in
                      the same rollouts, and the human managers, as reference
                      rows.

  results_cost.csv    WHAT IT SPENT. Targeting and restraint come apart: a
                      language-model manager run against this same clone
                      targets better than the human clone and then overpays,
                      and another pays for punishment and buys nothing. So a
                      seed that targets is not yet a seed that helps. Mean
                      punishment, common good, and the RL group's payoff
                      against the artificial punisher's payoff in the very same
                      episodes -- a paired within-run comparison of spend
                      against return, which needs no separate baseline run.

  results_noise.csv   Whether the mechanism was live: the adapted scale and the
                      divergence it held, over update steps, per seed.

Late window rather than the last point alone: one evaluation rollout is 1000
episodes but a single point in training, and the policy moves between points.
`--window` sets how many trailing evaluation points are pooled.

**On intervals.** Nothing here bootstraps over episodes. A symmetric control
elsewhere in this campaign disagreed with an independent measurement of the
same quantity at similar episode counts, with non-overlapping intervals, and
the live hypothesis is that episodes within a rollout are correlated enough
that bootstrap intervals over episodes are too narrow. The spread reported
here is across SEEDS and across evaluation points, which does not rest on that
assumption. Where a within-run spread appears it is labelled as such.

Usage:
    python scripts/rl_param_noise/results.py \\
        artifacts/manager/rl_pnoise_s4*/metrics/*.parquet \\
        --out plots/data_analysis/evaluation/rl_manager_param_noise
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aimanager.evaluation_suite.convert import (  # noqa: E402
    HUMAN_DATA_FILE,
    load_human,
)
from aimanager.evaluation_suite.metrics import (  # noqa: E402
    RPA_LABELS,
    ResponseMetrics,
)
from rl_param_noise.guard_report import md, weighted_profile, wide  # noqa: E402
from rl_param_noise.targeting import targeting  # noqa: E402

EVAL_TAG = "greedy"

COST_COLS = [
    "punishment",
    "common_good",
    "contribution",
    "group_payoff_sum",
    "opp_sum_payoff",
    "opp_punishment",
    "rl_avg_group_size",
    "opp_avg_group_size",
]


def late(df, window):
    """The trailing `window` evaluation points of each job."""
    keep = []
    for job, sub in df.groupby("job_id"):
        steps = sorted(sub[sub["sampling"] == EVAL_TAG]["update_step"].unique())
        keep.append(sub[sub["update_step"].isin(steps[-window:])])
    return pd.concat(keep, ignore_index=True)


def shape_table(df, window):
    ev = late(df, window)
    ev = ev[ev["sampling"] == EVAL_TAG]
    rows = {}
    for job in sorted(ev["job_id"].unique()):
        rows[job] = weighted_profile(ev[ev["job_id"] == job], "rpa")
    rows["artificial punisher (clone)"] = weighted_profile(ev, "rpa_opp")

    human = load_human(os.path.join(ROOT, HUMAN_DATA_FILE))
    h = ResponseMetrics().rpa(human)
    hm = h.groupby(level=0).mean().reindex(RPA_LABELS)
    hn = h.groupby(level=0).size().reindex(RPA_LABELS)
    rows["human managers"] = {
        **{k: float(hm[k]) for k in RPA_LABELS},
        **{f"n[{k}]": int(hn[k]) for k in RPA_LABELS},
    }

    shape = pd.DataFrame(rows).T
    shape.index.name = "manager"
    stats = pd.DataFrame(
        {
            name: targeting(
                [row[k] for k in RPA_LABELS], [row[f"n[{k}]"] for k in RPA_LABELS]
            )
            for name, row in shape.iterrows()
        }
    ).T
    # Spread of each seed's own contrast across the pooled evaluation points,
    # so "this seed is stable" and "these seeds agree" stay separate questions.
    spread = {}
    for job in sorted(ev["job_id"].unique()):
        sub = ev[ev["job_id"] == job]
        per_step = sub.groupby("update_step")[
            [f"rpa_mean_{k}" for k in RPA_LABELS]
        ].mean()
        spread[job] = float(
            (
                per_step[f"rpa_mean_{RPA_LABELS[0]}"]
                - per_step[f"rpa_mean_{RPA_LABELS[-1]}"]
            ).std()
        )
    shape = shape.join(stats)
    shape["contrast_sd_across_eval_points"] = pd.Series(spread)
    return shape


def cost_table(df, window):
    ev = late(df, window)
    ev = ev[ev["sampling"] == EVAL_TAG]
    rows = []
    for job in sorted(ev["job_id"].unique()):
        sub = ev[ev["job_id"] == job]
        row = {"manager": job}
        for c in COST_COLS:
            row[c] = float(sub[c].mean()) if c in sub else float("nan")
        # The paired within-run comparison: the RL manager's group against the
        # artificial punisher's, same episodes, same contributors.
        #
        # PER MEMBER, not summed. `group_payoff_sum` is a sum over whoever is
        # in the group, and group membership is endogenous -- members switch in
        # response to punishment -- so a manager that simply retains more
        # players scores higher on the sum without anyone being better off.
        # The summed difference is kept for continuity and must not be read as
        # welfare.
        row["payoff_per_member"] = row["group_payoff_sum"] / row["rl_avg_group_size"]
        row["opp_payoff_per_member"] = row["opp_sum_payoff"] / row["opp_avg_group_size"]
        row["payoff_per_member_minus_clone"] = (
            row["payoff_per_member"] - row["opp_payoff_per_member"]
        )
        row["payoff_sum_minus_clone"] = row["group_payoff_sum"] - row["opp_sum_payoff"]
        row["punishment_minus_clone"] = row["punishment"] - row["opp_punishment"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("manager")


def noise_table(df):
    cols = [c for c in df.columns if c.startswith("param_noise_")]
    if not cols:
        return None
    bh = df[df["sampling"] == "param-noise"]
    if bh.empty:
        return None
    return bh.groupby(["job_id", "update_step"])[cols].mean().reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = wide(args.parquets)
    shape = shape_table(df, args.window)
    cost = cost_table(df, args.window)
    shape.to_csv(os.path.join(args.out, "results_shape.csv"))
    cost.to_csv(os.path.join(args.out, "results_cost.csv"))

    n_eval = len(df[df["sampling"] == EVAL_TAG]["update_step"].unique())
    lines = [
        "# Parameter-space noise: results",
        "",
        f"Evaluated policy, pooled over the last {args.window} of {n_eval} "
        "evaluation points.",
        "",
        "## Did the shape recover",
        "",
        md(shape[list(RPA_LABELS)]),
        "",
        md(
            shape[
                [
                    "verdict",
                    "verdict_shape_only",
                    "tie_attenuated",
                    "monotonicity",
                    "rho_contribution_punishment",
                    "tau_b",
                    "n_distinct_bins",
                    "n_zero_bins",
                    "relative_range",
                    "contrast",
                    "contrast_sd_across_eval_points",
                ]
            ]
        ),
        "",
        "Row counts per bin:",
        "",
        md(shape[[f"n[{k}]" for k in RPA_LABELS]], "{:.0f}"),
        "",
        "## What it spent",
        "",
        "Paired within-run comparison against the artificial punisher, same",
        "episodes, same contributors, so no separate baseline run is needed.",
        "",
        "Read `payoff_per_member_minus_clone`, NOT the summed version.",
        "`group_payoff_sum` sums over whoever is in the group and membership",
        "is endogenous -- players switch in response to punishment -- so a",
        "manager that merely retains more members scores higher on the sum",
        "without anyone being better off. `common_good` is the per-capita",
        "pool and is the cleanest welfare number here.",
        "",
        "Targeting and restraint come apart: a seed that targets well and",
        "overpays is not a seed that helps.",
        "",
        md(cost),
    ]
    noise = noise_table(df)
    if noise is not None:
        noise.to_csv(os.path.join(args.out, "results_noise.csv"), index=False)
        last = noise.groupby("job_id").tail(1).set_index("job_id")
        lines += [
            "",
            "## Was the mechanism live",
            "",
            "Final adapted scale and the divergence it held, per seed. A scale",
            "at the cap with the divergence below target means weight noise",
            "could not match epsilon-greedy's displacement and the arm",
            "under-explored.",
            "",
            md(
                last[
                    [
                        "param_noise_scale",
                        "param_noise_target",
                        "param_noise_divergence",
                        "param_noise_divergence_l2",
                    ]
                ],
                "{:.4f}",
            ),
        ]

    path = os.path.join(args.out, "results.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
