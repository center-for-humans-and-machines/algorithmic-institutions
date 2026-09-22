"""Read the matched guard pilots and report what this arm is meant to move.

Two 200-step training runs, identical but for the exploration mechanism
(`rl_epsgreedy_guard` and `rl_pnoise_guard`, both seed 42), read from the
metrics parquets the training loop writes. Two outputs:

  guard_gap.md     THE MECHANISM. Mean punishment in the behaviour rollouts
                   against mean punishment in the evaluation rollout at the
                   same update steps, and their ratio. On the finished runs
                   the behaviour policy punished 1.7 to 6.6 times as hard as
                   the policy being evaluated; the claim of this arm is that
                   weight noise shrinks that.

  guard_shape.csv  THE OUTCOME. Mean punishment per contribution bin, on the
                   evaluation suite's own RPA bins, for each run and each
                   sampling, with the row count per bin. Beside them: the
                   artificial punisher (the clone of a human manager),
                   measured on the opponent's group in the very same
                   rollouts, and the human managers from
                   experiments/2group_8agent_50ep.csv.

  guard_noise.csv  This arm's own diagnostics: the adapted noise scale and
                   the divergence it held, per update step, under all three
                   measures.

Usage (local):
    python scripts/rl_param_noise/guard_report.py \\
        artifacts/manager/rl_epsgreedy_guard/metrics/rl_epsgreedy_guard.parquet \\
        artifacts/manager/rl_pnoise_guard/metrics/rl_pnoise_guard.parquet \\
        --out plots/data_analysis/evaluation/rl_manager_param_noise
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from aimanager.evaluation_suite.convert import (  # noqa: E402
    HUMAN_DATA_FILE,
    load_human,
)
from aimanager.evaluation_suite.metrics import (  # noqa: E402
    RPA_LABELS,
    ResponseMetrics,
)

EVAL_TAG = "greedy"


def wide(paths):
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df.pivot_table(
        index=["job_id", "sampling", "update_step", "round_number"],
        columns="metric",
        values="value",
    ).reset_index()


def weighted_profile(df, prefix):
    """Mean punishment per bin, weighting each recorded rollout-round mean by
    the number of agent-rounds behind it -- identical to pooling the raw rows."""
    out = {}
    for label in RPA_LABELS:
        n = df.get(f"{prefix}_n_{label}")
        v = df.get(f"{prefix}_mean_{label}")
        if n is None or v is None:
            out[label], out[f"n[{label}]"] = float("nan"), 0
            continue
        ok = n.notna() & v.notna() & (n > 0)
        total = float(n[ok].sum())
        out[label] = float((v[ok] * n[ok]).sum() / total) if total else float("nan")
        out[f"n[{label}]"] = int(total)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = wide(args.parquets)
    behaviour_tags = sorted(set(df["sampling"]) - {EVAL_TAG})

    # -- the gap ------------------------------------------------------
    lines = [
        "# Behaviour versus evaluated, 200-step guard pilots",
        "",
        "Mean punishment served by the RL manager to its own group. The",
        "behaviour rows are the rollouts that fill the replay buffer; the",
        "evaluated rows are the fully deterministic rollout at the same",
        "update steps, every exploration mechanism off. The ratio is the",
        "quantity the exploration comparison is about.",
        "",
        "| run | sampling | behaviour | evaluated | ratio | update steps |",
        "|---|---|---|---|---|---|",
    ]
    gap_rows = []
    for job in sorted(df["job_id"].unique()):
        sub = df[df["job_id"] == job]
        ev = sub[sub["sampling"] == EVAL_TAG]
        for tag in behaviour_tags:
            bh = sub[sub["sampling"] == tag]
            if bh.empty or ev.empty:
                continue
            steps = sorted(set(bh["update_step"]) & set(ev["update_step"]))
            b = float(bh[bh["update_step"].isin(steps)]["punishment"].mean())
            e = float(ev[ev["update_step"].isin(steps)]["punishment"].mean())
            ratio = b / e if e else float("nan")
            lines.append(
                f"| {job} | {tag} | {b:.4f} | {e:.4f} | {ratio:.3f} | {len(steps)} |"
            )
            gap_rows.append(
                {
                    "job_id": job,
                    "sampling": tag,
                    "behaviour": b,
                    "evaluated": e,
                    "ratio": ratio,
                    "n_update_steps": len(steps),
                }
            )
    pd.DataFrame(gap_rows).to_csv(os.path.join(args.out, "guard_gap.csv"), index=False)

    # -- the shape ----------------------------------------------------
    rows = {}
    for job in sorted(df["job_id"].unique()):
        for tag in sorted(df["sampling"].unique()):
            sub = df[(df["job_id"] == job) & (df["sampling"] == tag)]
            if sub.empty:
                continue
            rows[f"{job} [{tag}]"] = weighted_profile(sub, "rpa")
    # The artificial punisher, on the opponent's group in the same rollouts.
    ev_all = df[df["sampling"] == EVAL_TAG]
    if not ev_all.empty:
        rows["artificial punisher (clone)"] = weighted_profile(ev_all, "rpa_opp")

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
    shape = shape[list(RPA_LABELS) + [f"n[{k}]" for k in RPA_LABELS]]
    shape["contrast_0_minus_20"] = shape["{0}"] - shape["{20}"]
    shape.to_csv(os.path.join(args.out, "guard_shape.csv"))

    lines += [
        "",
        "## Policy shape",
        "",
        "Mean punishment per contribution bin, evaluation-suite RPA bins.",
        "Human managers fall from 4.76 at contribution 0 to 0.27 at 20; two",
        "of the three finished learned seeds rose instead. `contrast` is",
        "{0} minus {20}: positive is the human sign.",
        "",
        shape.to_markdown(floatfmt=".3f"),
    ]

    # -- this arm's diagnostics ---------------------------------------
    noise_cols = [c for c in df.columns if c.startswith("param_noise_")]
    if noise_cols:
        noise = (
            df[df["sampling"] == "param-noise"]
            .groupby(["job_id", "update_step"])[noise_cols]
            .mean()
            .reset_index()
        )
        noise.to_csv(os.path.join(args.out, "guard_noise.csv"), index=False)
        last = noise.tail(1).iloc[0]
        lines += [
            "",
            "## Noise scale",
            "",
            f"Final scale {last['param_noise_scale']:.5f}, holding divergence "
            f"{last['param_noise_divergence']:.3f} punishment levels "
            f"(l2 {last['param_noise_divergence_l2']:.5f}, "
            f"w1 {last['param_noise_divergence_w1']:.3f}).",
        ]

    path = os.path.join(args.out, "guard_gap.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
