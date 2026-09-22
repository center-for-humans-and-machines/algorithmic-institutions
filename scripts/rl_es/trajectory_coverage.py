"""Trajectory coverage: how far the states the behaviour policy visits sit
from the states the evaluated policy visits, per arm.

WHY THIS AND NOT THE PER-BIN SHIFT. DQN is off-policy. A behaviour policy that
differs from the target is what the algorithm is FOR: Q-learning bootstraps
toward the max over actions and evaluates the greedy policy whatever collected
the data. So the mere existence of a behaviour-versus-evaluated gap is not a
defect, and the per-bin punishment shift describes what was SAMPLED, not what
was learned.

What off-policy correction does not buy is states. It fixes the action choice
given a state; it cannot supply states the behaviour policy never visits. And
in this environment the state distribution is endogenous to the manager's own
behaviour, through two channels that are both in the code:

  * the contribution model is recurrent (`GraphNetwork.rnn_n`, hidden state
    carried across the 24 rounds), so a punishment at round 3 moves every
    later round's contributions through a hidden state, not just round 4's;
  * group composition is a function of punishment, through the switch
    predictor, so who the manager is even paid on at round 20 depends on what
    it did at round 3.

So what a CONSISTENTLY CONTINGENT manager produces over 24 rounds is a
trajectory that dithering does not generate cleanly, and a value function
trained on dithered data never sees the return of a coherent contingent
policy. Evolution strategies has no value function and no one-step backup: it
scores whole-episode returns of fixed policies, which is exactly the object
the argument says is missing.

WHAT THIS SCRIPT MEASURES, AND WHAT IT DOES NOT. Both arms log per-round
state summaries for a behaviour rollout and for an evaluation rollout at the
same update_step. This computes, per round, the gap between the two on each
summary, and reduces it to a per-episode trajectory distance.

  MEASURED: the divergence between the two rollouts' per-round MEAN state
    summaries -- mean contribution, the RL group's size, the common good.
  NOT MEASURED: the divergence between the two STATE DISTRIBUTIONS. These are
    batch means over 1000 episodes, so they say nothing about the spread, the
    joint distribution, or the parts of the state space either rollout reaches
    rarely. A small distance here is consistent with very different coverage.
  NOT MEASURED: that trajectory coverage is what causes any difference in what
    is learned. This is a description of the data each arm trained on.

Usage:
    python scripts/rl_es/trajectory_coverage.py \\
        --dqn <dir with rl_new_clones_s*.parquet> --seeds 42,43,44,45,46
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/evaluation/rl_manager_es")

#: The per-round state summaries both arms log. `punishment` is the ACTION,
#: not a state, and is reported separately for reference rather than folded
#: into the state distance.
STATE_METRICS = ["contribution", "rl_avg_group_size", "common_good"]
ACTION_METRIC = "punishment"
#: Whatever each arm calls its behaviour rollout.
BEHAVIOUR_TAGS = ("eps-greedy", "es-population")


def load(path, label):
    df = pd.read_parquet(path)
    # A directory may hold parquets in other schemas -- a simulation's
    # per_round file, another arm's diagnostics. Skip them by shape rather
    # than crashing on the first one.
    required = {"metric", "value", "sampling", "update_step", "round_number"}
    if not required.issubset(df.columns):
        print(f"skipping {path}: not a training metrics parquet")
        return None
    keep = STATE_METRICS + [ACTION_METRIC]
    df = df[df["metric"].isin(keep)]
    df = df.pivot_table(
        index=["update_step", "round_number", "sampling"],
        columns="metric",
        values="value",
    ).reset_index()
    df["run"] = label
    df["phase"] = np.where(
        df["sampling"].isin(BEHAVIOUR_TAGS), "behaviour", "evaluated"
    )
    return df


def diverge(df, label):
    """Per update_step: how far the behaviour trajectory sits from the
    evaluated one, in units of that run's own across-round variation."""
    rows = []
    wide = df.pivot_table(
        index=["update_step", "round_number"],
        columns="phase",
        values=STATE_METRICS + [ACTION_METRIC],
    )
    for step, frame in wide.groupby(level=0):
        row = {"run": label, "update_step": int(step)}
        for metric in STATE_METRICS + [ACTION_METRIC]:
            if (metric, "behaviour") not in frame or (metric, "evaluated") not in frame:
                continue
            b = frame[(metric, "behaviour")].to_numpy(dtype=float)
            e = frame[(metric, "evaluated")].to_numpy(dtype=float)
            row[f"{metric}_gap"] = float(np.nanmean(np.abs(b - e)))
            # Scaled by the evaluated trajectory's own across-round spread, so
            # metrics on different scales can be added up.
            spread = float(np.nanstd(e))
            row[f"{metric}_gap_scaled"] = (
                row[f"{metric}_gap"] / spread if spread > 1e-9 else np.nan
            )
        gaps = [row.get(f"{m}_gap_scaled") for m in STATE_METRICS]
        gaps = [g for g in gaps if g is not None and not np.isnan(g)]
        row["state_trajectory_distance"] = float(np.mean(gaps)) if gaps else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dqn", default=None, help="directory of DQN metrics parquets")
    ap.add_argument("--es", default=None, help="directory of ES metrics parquets")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sources = {}
    if args.dqn:
        for path in sorted(glob.glob(os.path.join(args.dqn, "*.parquet"))):
            sources[f"dqn:{os.path.basename(path)[:-8]}"] = path
    es_dir = args.es
    for seed in [int(s) for s in args.seeds.split(",")]:
        path = (
            os.path.join(es_dir, f"rl_es_s{seed}.parquet")
            if es_dir
            else os.path.join(
                ROOT, f"artifacts/manager/rl_es_s{seed}/metrics/rl_es_s{seed}.parquet"
            )
        )
        if os.path.exists(path):
            sources[f"es:rl_es_s{seed}"] = path

    frames, summaries = [], []
    for label, path in sources.items():
        df = load(path, label)
        if df is None:
            continue
        frames.append(df)
        summaries.append(diverge(df, label))
    if not frames:
        print("no metrics parquets found")
        return

    per_step = pd.concat(summaries)
    per_step.to_csv(
        os.path.join(args.out, "trajectory_coverage_per_step.csv"), index=False
    )
    # The late-training read is the one that matters: early on both arms sit
    # near the initialisation and nothing has diverged yet.
    late = per_step[
        per_step["update_step"]
        >= per_step.groupby("run")["update_step"].transform("max") * 0.5
    ]
    summary = (
        late.groupby("run")
        .agg(
            n_steps=("update_step", "size"),
            state_trajectory_distance=("state_trajectory_distance", "mean"),
            contribution_gap=("contribution_gap", "mean"),
            group_size_gap=("rl_avg_group_size_gap", "mean"),
            common_good_gap=("common_good_gap", "mean"),
            punishment_gap=("punishment_gap", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(args.out, "trajectory_coverage.csv"), index=False)
    print(summary.round(4).to_string(index=False))
    print(
        "\nMEASURED: divergence of per-round MEAN state summaries between each "
        "arm's behaviour and evaluation rollouts, over the second half of "
        "training.\nNOT MEASURED: state-distribution divergence, and any "
        "causal claim about what that divergence does to what is learned."
    )


if __name__ == "__main__":
    main()
