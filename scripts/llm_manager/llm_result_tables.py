"""The language-model arm beside every baseline it was run against.

`run_battery.py` contrasts each arm against the two bars its docstring names,
never-punishing and the incumbent `thr9_p10`. A language-model result also
has to be read against the manager clone it actually played and against the
capped sigmoid rule of #219, which is the best manager this project has, so
this regenerates the contrast table over all four references from the saved
per-episode frame. It is the same `battery.contrasts` on the same episodes,
called with more reference arms -- nothing is recomputed from the rollouts.

**Pool first, contribution beside it, never averaged.** The pool is the
settled objective and it separates good management from bad: the capped rule
gains 5 points there. But it does not reward punishment for its own sake --
the correct threshold rule beats never-punishing by about 0.5 on the pool,
which needs some 117,664 episodes to detect, so no budget here resolves it.
An arm level with never-punishing on the pool therefore has NOT been shown
to fail. What separates the two routes to the same pool is the contribution
column: one manager raised collaboration and paid for it, the other did
nothing.

Each run's own minimum detectable difference is computed from its own
per-episode spread rather than quoted from another run's table, because the
spread is a property of the world a manager makes.

    PYTHONPATH=src python scripts/llm_manager/llm_result_tables.py \\
        --run qwen3_8b=plots/.../qwen3-8b --run qwen3_32b=plots/.../qwen3-32b \\
        --out plots/data_analysis/llm_manager_battery
"""

import argparse
import os

import numpy as np
import pandas as pd

from aimanager.llm_manager import battery as bat

#: The comparison set, fixed before the run. The clone is both a baseline and
#: the rival every focal arm faced.
REFERENCES = ("never", "thr9_p10", "capped_sigmoid", "clone")


def contrasts_all(episodes, arms=REFERENCES):
    present = set(episodes["arm"])
    return pd.concat(
        [bat.contrasts(episodes, ref) for ref in arms if ref in present],
        ignore_index=True,
    )


def mdd_for(episodes, arm, quantity, n):
    """This arm's own minimum detectable difference at `n` episodes each.

    Unpaired, two-sided, 80% power, on the arm's own per-episode spread --
    not on a spread borrowed from a control that made a different world.
    """
    sd = float(np.nanstd(episodes.loc[episodes["arm"] == arm, f"focal_{quantity}"]))
    return 2.8016 * sd * np.sqrt(2.0 / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="a run directory written by run_battery.py; repeatable",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    headline, all_con = [], []
    for spec in args.run:
        name, _, path = spec.partition("=")
        ep = pd.read_parquet(os.path.join(path, "episodes.parquet"))
        bt = pd.read_csv(os.path.join(path, "battery.csv"))
        llm_arm = next(a for a in ep["arm"].unique() if str(a).startswith("llm"))

        con = contrasts_all(ep)
        con.insert(0, "run", name)
        con.to_csv(os.path.join(path, "contrasts_all_references.csv"), index=False)
        all_con.append(con)

        n_per_arm = int(ep.groupby("arm").size().min())
        for q in ("pool", "contribution"):
            for ref in REFERENCES:
                row = con[
                    (con["arm"] == llm_arm)
                    & (con["reference"] == ref)
                    & (con["quantity"] == q)
                ]
                if not len(row):
                    continue
                r = row.iloc[0]
                headline.append(
                    {
                        "run": name,
                        "arm": llm_arm,
                        "quantity": q,
                        "reference": ref,
                        "delta": r["delta"],
                        "lo": r["lo"],
                        "hi": r["hi"],
                        "crosses_zero": r["crosses_zero"],
                        "n_episodes_per_arm": n_per_arm,
                        "mdd_own_sd": mdd_for(ep, llm_arm, q, n_per_arm),
                    }
                )
        lvl = bt[bt["arm"] == llm_arm]
        print(f"\n=== {name}: {llm_arm}, {n_per_arm} episodes per arm ===")
        print(
            bt[["arm", "focal_pool", "focal_contribution", "focal_members"]]
            .sort_values("focal_pool", ascending=False)
            .to_string(index=False, float_format=lambda v: f"{v:.2f}")
        )
        if len(lvl):
            print(
                f"  parse_failure_rate {lvl['parse_failure_rate'].iloc[0]}  "
                f"total_tokens {lvl['total_tokens'].iloc[0]:.0f}"
            )

    head = pd.DataFrame(headline)
    head.to_csv(os.path.join(args.out, "llm_vs_baselines.csv"), index=False)
    pd.concat(all_con, ignore_index=True).to_csv(
        os.path.join(args.out, "contrasts_all_references.csv"), index=False
    )
    print("\n=== the language model against each baseline, pool first ===")
    for q in ("pool", "contribution"):
        print(f"\n-- {q} --")
        print(
            head[head["quantity"] == q]
            .drop(columns=["quantity", "arm"])
            .to_string(index=False, float_format=lambda v: f"{v:.2f}")
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
