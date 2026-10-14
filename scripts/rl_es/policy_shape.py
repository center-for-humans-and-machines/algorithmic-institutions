"""Policy shape of the evolution-strategies arm: mean punishment per
contribution bin, per seed, beside the human and clone columns.

This is the arm's PRIMARY outcome. The three finished DQN seeds punish, and
two of the three punish the wrong people -- monotone INCREASING in the
contribution, where human managers are monotone decreasing (4.76 at
contribution 0 down to 0.27 at 20). Uniform epsilon-greedy over 31 punishment
levels applies punishment independently of the contribution it is aimed at, so
it decorrelates the two in the replay buffer; this arm has no action noise at
all, so if that decorrelation is what produced the inverted shape, this arm is
where the human sign should come back.

Bins are the evaluation suite's own RPA edges, imported rather than restated,
so the columns here and
plots/data_analysis/evaluation/rl_manager_two_worlds/policy_shape.csv are the
same bins.

Two tables:
  policy_shape.csv       theta (the evaluated policy) per seed, at the last
                         evaluated generation, beside the human column and
                         whatever reference columns are supplied
  policy_shape_n.csv     the agent-rounds behind each of those means
  member_shape.csv       the same read for every population member of the
                         last generation, and the sign counts -- how much of
                         the spread is decided before selection
  shape_trajectory.csv   theta's per-bin means over the generations, so the
                         shape can be watched forming

CAVEAT, and it belongs in any write-up. These numbers come from the training
run's own evaluation rollouts (the mean parameter vector, deterministic, batch
1000, against the same opponent), not from the cross-evaluation simulation the
two-worlds numbers come from. The pairing and the episode count differ. Use
this to read the shape early and to compare ES seeds with each other; run the
cross-evaluation simulation before comparing a number here with a number
there.

Usage:
    python scripts/rl_es/policy_shape.py --seeds 42,43,44,45,46 \\
        --reference plots/data_analysis/evaluation/\\
            rl_manager_two_worlds/policy_shape.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.evaluation_suite.convert import (  # noqa: E402
    HUMAN_DATA_FILE,
    load_human,
)
from aimanager.evaluation_suite.metrics import (  # noqa: E402
    RPA_LABELS,
    ResponseMetrics,
)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/evaluation/rl_manager_es")


def human_column():
    human = load_human(os.path.join(ROOT, HUMAN_DATA_FILE))
    rpa = ResponseMetrics().rpa(human)
    return (
        rpa.groupby(level=0).mean().reindex(RPA_LABELS),
        rpa.groupby(level=0).size().reindex(RPA_LABELS),
    )


def load_shape(seed):
    path = os.path.join(
        ROOT,
        f"artifacts/manager/rl_es_s{seed}/metrics/rl_es_s{seed}_policy_shape.parquet",
    )
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def slope(series):
    """Least-squares slope of mean punishment across the six bins.

    Negative is the human sign: punish the free rider, leave the full
    contributor alone.
    """
    y = series.reindex(RPA_LABELS).to_numpy(dtype=float)
    x = np.arange(len(RPA_LABELS), dtype=float)
    ok = ~np.isnan(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--subset", default="all", choices=("all", "valid"))
    ap.add_argument(
        "--reference",
        default=None,
        help="a policy_shape.csv whose columns are prepended (e.g. the DQN arm's)",
    )
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    means, counts = {}, {}
    if args.reference and os.path.exists(args.reference):
        ref = pd.read_csv(args.reference, index_col=0).reindex(RPA_LABELS)
        for col in ref.columns:
            means[col] = ref[col]
    else:
        hm, hn = human_column()
        means["human managers"] = hm
        counts["human managers"] = hn

    members, trajectory, missing = [], [], []
    for seed in seeds:
        df = load_shape(seed)
        if df is None:
            missing.append(seed)
            continue
        df = df[df["subset"] == args.subset]
        theta = df[df["sampling"] == "greedy"]
        last = int(theta["update_step"].max())
        final = theta[theta["update_step"] == last].set_index("contribution_bin")
        means[f"es_s{seed}"] = final["mean_punishment"].reindex(RPA_LABELS)
        counts[f"es_s{seed}"] = final["n"].reindex(RPA_LABELS)

        traj = theta.pivot_table(
            index="update_step", columns="contribution_bin", values="mean_punishment"
        ).reindex(columns=RPA_LABELS)
        traj["seed"] = seed
        traj["slope"] = [slope(traj.loc[i, RPA_LABELS]) for i in traj.index]
        trajectory.append(traj.reset_index())

        pop = df[(df["sampling"] == "es-population") & (df["member"] >= 0)]
        pop = pop[pop["update_step"] == int(pop["update_step"].max())]
        table = pop.pivot_table(
            index="contribution_bin", columns="member", values="mean_punishment"
        ).reindex(RPA_LABELS)
        for member in table.columns:
            members.append(
                {
                    "seed": seed,
                    "generation": last,
                    "member": int(member),
                    "slope": slope(table[member]),
                    **{b: table[member].get(b) for b in RPA_LABELS},
                }
            )

    shape = pd.DataFrame(means).reindex(RPA_LABELS)
    shape.index.name = "contribution_bin"
    shape.to_csv(os.path.join(args.out, "policy_shape.csv"))
    if counts:
        cnt = pd.DataFrame(counts).reindex(RPA_LABELS)
        cnt.index.name = "contribution_bin"
        cnt.to_csv(os.path.join(args.out, "policy_shape_n.csv"))

    slopes = pd.Series({c: slope(shape[c]) for c in shape.columns}, name="slope")
    slopes.index.name = "manager"
    slopes.to_frame().assign(
        sign=lambda d: np.where(
            d["slope"] < 0, "human (decreasing)", "inverted (increasing)"
        )
    ).to_csv(os.path.join(args.out, "policy_shape_slope.csv"))

    if members:
        m = pd.DataFrame(members)
        m.to_csv(os.path.join(args.out, "member_shape.csv"), index=False)
        summary = (
            m.groupby("seed")["slope"]
            .agg(
                n_members="size",
                human_sign=lambda s: int((s < 0).sum()),
                inverted_sign=lambda s: int((s > 0).sum()),
                slope_min="min",
                slope_max="max",
                slope_sd="std",
            )
            .reset_index()
        )
        summary.to_csv(os.path.join(args.out, "member_shape_summary.csv"), index=False)
        print(summary.to_string(index=False))
    if trajectory:
        pd.concat(trajectory).to_csv(
            os.path.join(args.out, "shape_trajectory.csv"), index=False
        )

    print(shape.to_string())
    print(slopes.to_string())
    if missing:
        print(f"no policy shape yet for seeds: {missing}")


if __name__ == "__main__":
    main()
