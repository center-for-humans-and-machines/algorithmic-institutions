"""Who leaves: the contribution differential between leavers and stayers.

A mechanism-level read of a manager's targeting direction that needs no
counterfactual. At each decision round, among the players in the manager's own
group, compare what the ones who switch away contributed against what the ones
who stay contributed:

    differential = mean(contribution | leaves) - mean(contribution | stays)

NEGATIVE means the players leaving are the ones who contributed less: the
manager is shedding free-riders. POSITIVE means the leavers are the
contributors: the manager is **selecting for free-riders**, which is a
stronger statement than failing to discipline them, because the group
composition is actively moving the wrong way.

Reference values on this world, measured elsewhere: a correctly targeted rule
−3.51, this project's human clone −2.35, never-punishing −1.20, and every
deliberately inverted rule positive, up to +1.83.

The never-punishing value is not zero, and that matters for reading a
collapsed seed. Switching responds to more than punishment, so a manager that
punishes nothing still sits at −1.20 rather than 0. **A seed whose population
has collapsed to a flat or zero policy should land near −1.20, and its doing
so is a check that the collapse detector agrees with the behaviour** -- not
evidence that the manager targeted anything.

Usage:
    python scripts/rl_es/leaver_selection.py \\
        plots/simulation/25_rl_es_cross_eval/per_round.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.evaluation_suite.convert import load_sim  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/evaluation/rl_manager_es")


def short(name):
    for a, b in (
        ("ah group_switching managed by ", ""),
        ("_vs_punisher", ""),
        ("_self", ""),
    ):
        name = name.replace(a, b)
    return name


def differential(df, n_boot=2000, seed=42):
    """Leaver-minus-stayer contribution, with a bootstrap CI over episodes.

    Resampled by EPISODE rather than by row: rows within an episode share a
    trajectory and a group composition, so a row bootstrap would understate
    the interval badly.
    """
    d = df[df["group_id"] == 0]
    d = d[d["switch_valid"]].dropna(subset=["contribution"])
    if d.empty:
        return {}
    leavers = d[d["does_switch"]]["contribution"]
    stayers = d[~d["does_switch"]]["contribution"]
    if len(leavers) == 0 or len(stayers) == 0:
        return {}
    point = float(leavers.mean() - stayers.mean())

    rng = np.random.default_rng(seed)
    episodes = d["episode_id"].unique()
    by_episode = {e: g for e, g in d.groupby("episode_id")}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(episodes, size=len(episodes), replace=True)
        s = pd.concat([by_episode[e] for e in pick])
        lv, st = (
            s[s["does_switch"]]["contribution"],
            s[~s["does_switch"]]["contribution"],
        )
        if len(lv) and len(st):
            draws.append(lv.mean() - st.mean())
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
    return {
        "differential": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_leavers": int(len(leavers)),
        "n_stayers": int(len(stayers)),
        "mean_contribution_leavers": float(leavers.mean()),
        "mean_contribution_stayers": float(stayers.mean()),
        "leave_rate": float(d["does_switch"].mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("per_round")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sims = {short(k): v for k, v in load_sim(args.per_round).items()}
    rows = []
    for name, df in sorted(sims.items()):
        stats = differential(df, n_boot=args.boot)
        if stats:
            rows.append({"manager": name, **stats})
    out = pd.DataFrame(rows).sort_values("differential")
    out.to_csv(os.path.join(args.out, "leaver_selection.csv"), index=False)

    print("=== WHO LEAVES: leaver-minus-stayer contribution, group 0 ===")
    print(
        out[
            [
                "manager",
                "differential",
                "ci_lo",
                "ci_hi",
                "mean_contribution_leavers",
                "mean_contribution_stayers",
                "leave_rate",
                "n_leavers",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(
        "\nNegative = the manager sheds free-riders. Positive = it sheds "
        "contributors, i.e. selects FOR free-riders.\n"
        "Reference: correctly targeted -3.51, human clone -2.35, "
        "never-punishing -1.20, inverted rules positive to +1.83."
    )


if __name__ == "__main__":
    main()
