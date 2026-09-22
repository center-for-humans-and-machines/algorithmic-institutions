"""Guard and read-out for the bootstrapped-DQN arm.

Reads one or more training metric parquets
(`artifacts/manager/<job>/metrics/<job>.parquet`) and writes, per job:

  gap.csv           mean punishment under the behaviour policy against mean
                    punishment under the evaluated policy, per evaluation
                    point, with their ratio. A DESCRIPTION OF WHAT WAS
                    SAMPLED, not a defect: DQN is off-policy and a behaviour
                    policy that differs from the target is what the algorithm
                    is for. A large ratio is not a fault and a small one is
                    not an improvement.
  policy_shape.csv  mean punishment per contribution bin for the evaluated
                    policy, on the evaluation suite's own RPA bins, beside the
                    human and clone columns measured on
                    auto/rl-manager-two-worlds.
  head_shape.csv    per-head slope (top bin minus bottom bin) and the ensemble
                    diversity diagnostics, per evaluation point. The headline
                    is whether the heads disagree about the SIGN.

    python scripts/rl_bootstrapped/guard.py <parquet> [<parquet> ...] \\
        [--out plots/data_analysis/evaluation/rl_manager_bootstrapped_dqn]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.evaluation_suite.metrics import RPA_LABELS  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/evaluation/rl_manager_bootstrapped_dqn")
# Measured on auto/rl-manager-two-worlds by scripts/rl_two_worlds/measure.py;
# copied here so this arm's table carries its reference columns without
# depending on that branch being checked out. Provenance in README.md.
REFERENCE = os.path.join(OUT, "reference_policy_shape.csv")

BEHAVIOUR_TAGS = ("bootstrap-head", "eps-greedy")
EVAL_TAG = "greedy"

# The shape-distortion prediction, shared with the annealed-local arm so the
# arms are comparable. Under epsilon-greedy a uniform resample over the 31
# punishment levels 0..30 has mean 15, so WITHIN EACH CONTRIBUTION BIN the
# behaviour policy is dragged toward 15 by exactly eps * (15 - evaluated).
# No free parameters. The reference eps is held at the historical 0.1 for
# every arm, including arms that dither less or not at all: the fitted slope
# then reads as "how much of the epsilon-greedy distortion this arm still
# has", 1 being all of it and 0 none.
UNIFORM_MEAN = 15.0
REFERENCE_EPS = 0.1


def _wide(df):
    """Long metrics frame -> one row per (sampling, update_step, round)."""
    return df.pivot_table(
        index=["sampling", "update_step", "round_number"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()


def gap_table(wide):
    """Behaviour against evaluated, averaged over the 24 rounds of a rollout.

    Both sides are the RL manager's own group only (`punishment` is masked to
    it upstream), so the ratio is the exploration overshoot and nothing else.
    """
    per_rollout = wide.groupby(["sampling", "update_step"])["punishment"].mean()
    per_rollout = per_rollout.unstack("sampling")
    behaviour = [c for c in per_rollout.columns if c in BEHAVIOUR_TAGS]
    if not behaviour or EVAL_TAG not in per_rollout.columns:
        return None
    out = pd.DataFrame(
        {
            "behaviour_punishment": per_rollout[behaviour[0]],
            "evaluated_punishment": per_rollout[EVAL_TAG],
        }
    ).dropna()
    out["gap"] = out["behaviour_punishment"] - out["evaluated_punishment"]
    out["ratio"] = out["behaviour_punishment"] / out["evaluated_punishment"].where(
        out["evaluated_punishment"] > 0
    )
    out["sampling"] = behaviour[0]
    return out


def shape_table(wide, sampling=EVAL_TAG, last_n=5):
    """Count-weighted pool of the per-round binned means over the last
    `last_n` evaluation points -- the same pooling `measure.py` does over a
    simulation's rows, recovered from the per-round sums the training loop
    logged."""
    d = wide[wide["sampling"] == sampling]
    if d.empty:
        return None
    steps = sorted(d["update_step"].unique())[-last_n:]
    d = d[d["update_step"].isin(steps)]
    rows = {}
    for i, label in enumerate(RPA_LABELS):
        mean, n = f"rpa_mean_b{i}", f"rpa_n_b{i}"
        if mean not in d.columns:
            return None
        ok = d[[mean, n]].dropna()
        total = (ok[mean] * ok[n]).sum()
        count = ok[n].sum()
        rows[label] = {
            "mean_punishment": total / count if count else float("nan"),
            "n_agent_rounds": int(count),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "contribution_bin"
    return out


def behaviour_tag(wide):
    present = set(wide["sampling"].unique())
    for tag in BEHAVIOUR_TAGS:
        if tag in present:
            return tag
    return None


def distortion_table(wide, last_n=5):
    """Per-bin shape distortion: how far the replay buffer's shape sits from
    the shape being evaluated, against the parameter-free epsilon-greedy
    prediction.

    A finer description of the sampled distribution than the aggregate gap,
    which only sees a level offset. It says whether the exploration mechanism
    flattens the punishment-contribution contingency *in the data it
    collects*.

    It does NOT say what was learned. Off-policy, a flattened sampling profile
    is not by itself a fault -- Q-learning evaluates the greedy policy
    whatever collected the data. The claim this arm actually rests on is
    trajectory coverage, which lives at the episode level and is not measured
    here; see notes/autoresearch_log/rl-manager-bootstrapped-dqn.md.
    """
    tag = behaviour_tag(wide)
    evaluated = shape_table(wide, EVAL_TAG, last_n)
    behaviour = shape_table(wide, tag, last_n) if tag else None
    if evaluated is None or behaviour is None:
        return None, None
    d = pd.DataFrame(
        {
            "evaluated": evaluated["mean_punishment"],
            "behaviour": behaviour["mean_punishment"],
            "n_evaluated": evaluated["n_agent_rounds"],
            "n_behaviour": behaviour["n_agent_rounds"],
        }
    ).reindex(RPA_LABELS)
    d["observed_shift"] = d["behaviour"] - d["evaluated"]
    d["predicted_shift"] = REFERENCE_EPS * (UNIFORM_MEAN - d["evaluated"])
    # The prediction changes sign exactly where the evaluated policy already
    # punishes above the uniform mean -- a bin where dithering pulls DOWN.
    d["predicted_sign"] = np.sign(d["predicted_shift"])
    d["observed_sign"] = np.sign(d["observed_shift"])

    ok = d.dropna(subset=["observed_shift", "predicted_shift"])
    summary = {"behaviour_sampling": tag, "n_bins": len(ok)}
    if len(ok) >= 2 and ok["predicted_shift"].std() > 0:
        slope, intercept = np.polyfit(ok["predicted_shift"], ok["observed_shift"], 1)
        summary["distortion_slope"] = float(slope)
        summary["distortion_intercept"] = float(intercept)
    summary["mean_abs_shift"] = float(ok["observed_shift"].abs().mean())
    w = ok["n_evaluated"]
    if w.sum() > 0:
        summary["mean_abs_shift_weighted"] = float(
            (ok["observed_shift"].abs() * w).sum() / w.sum()
        )
    summary["sign_agreement"] = float(
        (ok["predicted_sign"] == ok["observed_sign"]).mean()
    )
    # Flattening: how much shallower the replay buffer's slope is than the
    # evaluated policy's. Epsilon-greedy predicts exactly eps.
    ev_slope = d["evaluated"].iloc[-1] - d["evaluated"].iloc[0]
    beh_slope = d["behaviour"].iloc[-1] - d["behaviour"].iloc[0]
    summary["evaluated_slope"] = float(ev_slope)
    summary["behaviour_slope"] = float(beh_slope)
    if ev_slope == ev_slope and ev_slope != 0:
        summary["flattening"] = float(1.0 - beh_slope / ev_slope)
    return d, summary


def head_table(wide, sampling=EVAL_TAG):
    cols = [
        c for c in wide.columns if c.startswith("head_") or c.startswith("consensus_")
    ]
    if not cols:
        return None
    d = wide[wide["sampling"] == sampling]
    return d.groupby("update_step")[cols].mean()


def job_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--last-n", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    reference = None
    if os.path.exists(REFERENCE):
        reference = pd.read_csv(REFERENCE, index_col="contribution_bin")

    shapes, gaps, distortions = {}, [], []
    for path in args.parquets:
        job = job_name(path)
        wide = _wide(pd.read_parquet(path))
        job_out = os.path.join(args.out, job)
        os.makedirs(job_out, exist_ok=True)

        gap = gap_table(wide)
        if gap is not None:
            gap.to_csv(os.path.join(job_out, "gap.csv"))
            last = gap.iloc[-1]
            gaps.append(
                {
                    "job": job,
                    "behaviour_punishment": last["behaviour_punishment"],
                    "evaluated_punishment": last["evaluated_punishment"],
                    "gap": last["gap"],
                    "ratio": last["ratio"],
                    "n_eval_points": len(gap),
                }
            )

        shape = shape_table(wide, last_n=args.last_n)
        if shape is not None:
            shape.to_csv(os.path.join(job_out, "policy_shape.csv"))
            shapes[job] = shape["mean_punishment"]
            shapes[f"{job} (n)"] = shape["n_agent_rounds"]

        dist, summary = distortion_table(wide, last_n=args.last_n)
        if dist is not None:
            dist.to_csv(os.path.join(job_out, "shape_distortion.csv"))
            distortions.append({"job": job, **summary})

        heads = head_table(wide)
        if heads is not None:
            heads.to_csv(os.path.join(job_out, "head_shape.csv"))

    if distortions:
        ddf = pd.DataFrame(distortions).set_index("job")
        ddf.to_csv(os.path.join(args.out, "shape_distortion_summary.csv"))
        print("\n== shape distortion: behaviour vs evaluated, per bin ==")
        print("(distortion_slope 1 = full epsilon-greedy distortion, 0 = none)")
        print(ddf.to_string())

    if gaps:
        gdf = pd.DataFrame(gaps).set_index("job")
        gdf.to_csv(os.path.join(args.out, "gap_summary.csv"))
        print("\n== behaviour vs evaluated punishment (last eval point) ==")
        print(gdf.to_string())

    if shapes:
        table = pd.DataFrame(shapes)
        if reference is not None:
            keep = [c for c in ("human managers", "lin_punisher") if c in reference]
            table = reference[keep].join(table)
        table.index.name = "contribution_bin"
        table = table.reindex(RPA_LABELS)
        table.to_csv(os.path.join(args.out, "policy_shape_all.csv"))
        print("\n== mean punishment by contribution bin ==")
        print(table.to_string())


if __name__ == "__main__":
    main()
