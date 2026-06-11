#!/usr/bin/env python3
"""Plot training-time metrics for one or more RL manager runs.

Reads the long-format parquet at <artifact_dir>/metrics/<job_id>.parquet
(produced by ``src/aimanager/rl_manager.py``) and renders a grid of
metric-vs-update_step subplots, one line per run.

Aggregation per (run, update_step):
  - Default: mean across the 24 rounds of the eval/greedy rollout.
  - ``rl_group_size``: takes the LAST round only, mirroring the wandb
    eval/rl_group_size logging (captures end-of-episode group composition,
    not within-episode dynamics).

Example
-------
    python scripts/plotting/plot_rl_manager_metrics.py \\
        artifacts/manager/03_2g8a_avg \\
        artifacts/manager/03_2g8a_sum \\
        --output-dir plots/rl_manager/2g8a_sum_vs_avg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Metrics where the round-level aggregation is the LAST round, not the mean.
LAST_ROUND_METRICS = {"rl_end_group_size", "opp_end_group_size"}

# Default metric order for the subplot grid.
DEFAULT_METRICS = [
    "loss",
    "next_reward",
    "group_payoff",
    "group_payoff_sum",
    "opp_sum_payoff",
    "rl_end_group_size",
    "opp_end_group_size",
    "rl_avg_group_size",
    "opp_avg_group_size",
    "contribution",
    "punishment",
    "opp_punishment",
    "common_good",
    "contributor_payoff",
    "q_mean",
]


def load_run(artifact_dir: Path) -> pd.DataFrame:
    # Prefer <dir_name>.parquet, else fall back to whatever single parquet
    # lives in metrics/ (handles legacy runs where the parquet name doesn't
    # match the artifact directory name).
    candidate = artifact_dir / "metrics" / f"{artifact_dir.name}.parquet"
    if candidate.exists():
        parquet = candidate
    else:
        matches = list((artifact_dir / "metrics").glob("*.parquet"))
        if not matches:
            raise FileNotFoundError(
                f"No metrics parquet under {artifact_dir / 'metrics'}"
            )
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Multiple parquets in {artifact_dir / 'metrics'} — "
                f"specify the expected one as <dir>/{artifact_dir.name}.parquet"
            )
        parquet = matches[0]
    df = pd.read_parquet(parquet)
    df["run"] = artifact_dir.name
    return df


def aggregate(df: pd.DataFrame, sampling: str) -> pd.DataFrame:
    """Reduce (round_number, update_step) -> (update_step) per metric."""
    df = df[df["sampling"] == sampling].copy()
    last_round = df["round_number"].max()
    mean_rows = df[~df["metric"].isin(LAST_ROUND_METRICS)]
    last_rows = df[
        df["metric"].isin(LAST_ROUND_METRICS) & (df["round_number"] == last_round)
    ]
    agg_mean = (
        mean_rows.groupby(["run", "update_step", "metric"], as_index=False)["value"]
        .mean()
    )
    agg_last = last_rows[["run", "update_step", "metric", "value"]]
    return pd.concat([agg_mean, agg_last], ignore_index=True)


def plot_grid(
    agg: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    sampling: str,
) -> None:
    # `loss` appears in greedy rows as NaN (it's only computed on the
    # off-policy update). Drop any metric whose values are entirely null.
    non_null = (
        agg.groupby("metric")["value"].apply(lambda s: s.notna().any()).to_dict()
    )
    available = [m for m in metrics if non_null.get(m, False)]
    skipped = [m for m in metrics if not non_null.get(m, False)]
    if skipped:
        print(f"  ({sampling}) skipping metrics with no data: {skipped}")
    if not available:
        raise ValueError(f"None of {metrics} present in parquet metric column")

    n = len(available)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows), squeeze=False)

    for i, metric in enumerate(available):
        ax = axes[i // cols][i % cols]
        sub = agg[agg["metric"] == metric]
        for run, run_sub in sub.groupby("run"):
            run_sub = run_sub.sort_values("update_step")
            ax.plot(run_sub["update_step"], run_sub["value"], label=run, lw=1.4)
        ax.set_title(metric)
        ax.set_xlabel("update_step")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=9)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"RL manager metrics ({sampling})", y=1.005, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "artifact_dirs",
        nargs="+",
        type=Path,
        help="One or more artifact directories (e.g. artifacts/manager/03_2g8a_sum).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/rl_manager"),
        help="Destination directory (default: plots/rl_manager).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metric names to plot (default: full eval grid).",
    )
    parser.add_argument(
        "--sampling",
        choices=["greedy", "eps-greedy", "both"],
        default="greedy",
        help="Which rollout type to plot. 'both' emits two separate figures.",
    )
    args = parser.parse_args()

    frames = [load_run(d) for d in args.artifact_dirs]
    df = pd.concat(frames, ignore_index=True)

    samplings = ["greedy", "eps-greedy"] if args.sampling == "both" else [args.sampling]
    for sampling in samplings:
        agg = aggregate(df, sampling=sampling)
        if agg.empty:
            print(f"No rows for sampling={sampling!r}; skipping.")
            continue
        suffix = "_eps_greedy" if sampling == "eps-greedy" else ""
        out = args.output_dir / f"metrics{suffix}.png"
        plot_grid(agg, args.metrics, out, sampling=sampling)


if __name__ == "__main__":
    main()
