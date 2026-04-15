#!/usr/bin/env python3
"""Analyze AH training metrics (single run OR grid search).

Focus: **test log-loss** (the figure of merit for the generative model) plus
feature-importance via shuffle and ablation. Handles a single-run artifact
directory (one metrics parquet) or a grid-search directory (many parquets).

Example usage
-------------
    # Grid search (multi-parquet)
    python scripts/plotting/analyze_ah_metrics.py \
        artifacts/artificial_humans/switch_pred_grid_search

    # Single run
    python scripts/plotting/analyze_ah_metrics.py \
        artifacts/artificial_humans/switch_pred_mlp_rnn_feat

    # Use the last epoch instead of the best-test-loss epoch
    python scripts/plotting/analyze_ah_metrics.py <artifact_dir> \
        --epoch-mode last

Prints to stdout:
- Ranked config summary (test log-loss, accuracy, MAE).
- Per-config feature importance for shuffle and ablate.
- Wide pivot tables of feature importance across all runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


# Columns that are metadata, not hyper-parameters, when detecting configs.
RESERVED_COLS = {
    "name",
    "value",
    "cv_split",
    "epoch",
    "set",
    "strategy",
    "n_pred",
    "mask",
    "shuffle_feature",
    "ablate_feature",
    "job_id",
}


def load_metrics(path: Path) -> pd.DataFrame:
    """Load all metrics parquets under *path* (single file, metrics dir, or
    artifact dir containing a `metrics/` subfolder)."""
    if path.is_file():
        return pd.read_parquet(path)

    metrics_dir = path / "metrics" if (path / "metrics").exists() else path
    files = sorted(metrics_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No metrics parquets under {metrics_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def detect_config_cols(df: pd.DataFrame) -> List[str]:
    """Return the list of label columns that differ across rows and were
    emitted by the grid search (or single-run labels)."""
    candidates = [c for c in df.columns if c not in RESERVED_COLS]
    # Keep only columns that vary (so a single-run file reduces to []).
    return [c for c in candidates if df[c].nunique(dropna=False) > 1]


def filter_baseline(df: pd.DataFrame, mask: int = 0) -> pd.DataFrame:
    """Rows with no feature perturbation at a given mask pattern."""
    shuffle_na = (
        df["shuffle_feature"].isna()
        if "shuffle_feature" in df.columns
        else pd.Series(True, index=df.index)
    )
    ablate_na = (
        df["ablate_feature"].isna()
        if "ablate_feature" in df.columns
        else pd.Series(True, index=df.index)
    )
    return df[(df["mask"] == mask) & shuffle_na & ablate_na]


def pick_epoch_per_fold(
    baseline_test_ll: pd.DataFrame,
    config_cols: List[str],
    mode: str,
) -> pd.DataFrame:
    """For each (config, cv_split), return the chosen epoch.

    - mode='best': epoch minimising baseline test log-loss (per fold)
    - mode='last': the final epoch logged for that fold
    Returns columns: config_cols + [cv_split, epoch]
    """
    group_cols = config_cols + ["cv_split"]
    if mode == "best":
        idx = baseline_test_ll.groupby(group_cols, dropna=False)["value"].idxmin()
        chosen = baseline_test_ll.loc[idx, group_cols + ["epoch"]]
    elif mode == "last":
        idx = baseline_test_ll.groupby(group_cols, dropna=False)["epoch"].idxmax()
        chosen = baseline_test_ll.loc[idx, group_cols + ["epoch"]]
    else:
        raise ValueError(f"Unknown epoch mode: {mode}")
    return chosen.reset_index(drop=True)


def test_loss_summary(
    df: pd.DataFrame,
    chosen: pd.DataFrame,
    config_cols: List[str],
    mask: int = 0,
) -> pd.DataFrame:
    """Per-config summary: test log_loss, accuracy (greedy), MAE (greedy).

    Metrics are taken at the chosen epoch per fold, then aggregated.
    """
    join_cols = config_cols + ["cv_split", "epoch"]

    def _collect(name: str, strategy: Optional[str]) -> pd.Series:
        cond = (
            (df["set"] == "test")
            & (df["name"] == name)
            & (df["mask"] == mask)
        )
        if "shuffle_feature" in df.columns:
            cond &= df["shuffle_feature"].isna()
        if "ablate_feature" in df.columns:
            cond &= df["ablate_feature"].isna()
        sub = df[cond]
        if strategy is None:
            sub = sub[sub["strategy"].isna()]
        else:
            sub = sub[sub["strategy"] == strategy]
        merged = chosen.merge(sub, on=join_cols, how="left")
        return merged["value"]

    chosen = chosen.copy()
    chosen["test_log_loss"] = _collect("log_loss", None).values
    chosen["test_accuracy"] = _collect("accuracy", "greedy").values
    chosen["test_mae"] = _collect("mean_absolute_error", "greedy").values
    if not config_cols:
        chosen["_dummy"] = "all"

    agg = (
        chosen.groupby(config_cols if config_cols else ["_dummy"], dropna=False)
        .agg(
            test_log_loss_mean=("test_log_loss", "mean"),
            test_log_loss_std=("test_log_loss", "std"),
            test_log_loss_min=("test_log_loss", "min"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            n_folds=("test_log_loss", "count"),
            best_epoch_mean=("epoch", "mean"),
        )
        .reset_index()
        .sort_values("test_log_loss_mean")
    )
    if "_dummy" in agg.columns:
        agg = agg.drop(columns="_dummy")
    return agg


def feature_importance(
    df: pd.DataFrame,
    chosen: pd.DataFrame,
    config_cols: List[str],
    mask: int = 0,
) -> pd.DataFrame:
    """Delta-log-loss per config x feature for shuffle and ablate methods.

    Positive delta = perturbing the feature hurt the model = feature mattered.
    Evaluated at the chosen epoch of the *baseline* run (same epoch per fold).
    """
    join_cols = config_cols + ["cv_split", "epoch"]

    # Baseline test log-loss per (config, fold, epoch)
    base_cond = (
        (df["set"] == "test")
        & (df["name"] == "log_loss")
        & (df["mask"] == mask)
    )
    if "shuffle_feature" in df.columns:
        base_cond &= df["shuffle_feature"].isna()
    if "ablate_feature" in df.columns:
        base_cond &= df["ablate_feature"].isna()
    baseline = df[base_cond][join_cols + ["value"]].rename(
        columns={"value": "baseline_log_loss"}
    )

    # Baseline joined to chosen epochs
    base_at_chosen = chosen.merge(baseline, on=join_cols, how="left")

    long_rows = []
    for method, col in [("shuffle", "shuffle_feature"), ("ablate", "ablate_feature")]:
        if col not in df.columns or df[col].dropna().empty:
            continue
        perturbed = df[
            (df["set"] == "test")
            & (df["name"] == "log_loss")
            & (df["mask"] == mask)
            & df[col].notna()
        ][join_cols + [col, "value"]].rename(columns={"value": "perturbed_log_loss"})
        merged = base_at_chosen.merge(perturbed, on=join_cols, how="left")
        merged["delta"] = merged["perturbed_log_loss"] - merged["baseline_log_loss"]
        merged["feature"] = merged[col]
        merged["method"] = method
        long_rows.append(
            merged[config_cols + ["cv_split", "feature", "method", "delta"]]
        )

    if not long_rows:
        return pd.DataFrame(
            columns=config_cols + ["feature", "method", "delta_mean", "delta_std"]
        )

    long_df = pd.concat(long_rows, ignore_index=True)
    group_cols = (
        (config_cols if config_cols else ["_dummy"]) + ["feature", "method"]
    )
    if not config_cols:
        long_df["_dummy"] = "all"
    summary = (
        long_df.groupby(group_cols, dropna=False)
        .agg(
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            n_folds=("delta", "count"),
        )
        .reset_index()
        .sort_values(group_cols[:-2] + ["delta_mean"], ascending=[True] * len(
            group_cols[:-2]
        ) + [False])
    )
    if "_dummy" in summary.columns:
        summary = summary.drop(columns="_dummy")
    return summary


def pivot_feature_importance(
    fi: pd.DataFrame,
    summary: pd.DataFrame,
    config_cols: List[str],
    method: str,
) -> pd.DataFrame:
    """Wide table: rows = configs (ranked by test log-loss), cols = features.

    Values are delta_mean for the given method. A single-run input (no
    config columns) collapses to a single-row table.
    """
    sub = fi[fi["method"] == method].copy()
    if sub.empty:
        return pd.DataFrame()

    if config_cols:
        # Build a composite config label that mirrors the summary order.
        sub["config"] = sub[config_cols].astype(str).agg(" | ".join, axis=1)
        summary = summary.copy()
        summary["config"] = summary[config_cols].astype(str).agg(" | ".join, axis=1)
        config_order = summary["config"].tolist()
    else:
        sub["config"] = "all"
        config_order = ["all"]

    wide = (
        sub.pivot_table(
            index="config",
            columns="feature",
            values="delta_mean",
            aggfunc="mean",
        )
        .reindex(config_order)
    )
    # Put stronger (mean |delta|) features first for readability.
    feature_order = wide.abs().mean(axis=0).sort_values(ascending=False).index
    wide = wide[feature_order]

    # Attach test log-loss for cross-reference.
    if config_cols:
        loss_map = summary.set_index("config")["test_log_loss_mean"]
        wide.insert(0, "test_log_loss_mean", loss_map.reindex(wide.index))
    return wide


def print_feature_importance_table(
    fi: pd.DataFrame, summary: pd.DataFrame, config_cols: List[str]
) -> None:
    if fi.empty:
        return
    for method in sorted(fi["method"].unique()):
        wide = pivot_feature_importance(fi, summary, config_cols, method)
        if wide.empty:
            continue
        print(f"\n=== Feature importance table ({method}) ===")
        print("     rows = configs (sorted by test log-loss, best first)")
        print("     cols = features (sorted by |delta|, strongest first)")
        fmt = {c: "{:+.4f}".format for c in wide.columns}
        if "test_log_loss_mean" in wide.columns:
            fmt["test_log_loss_mean"] = "{:.4f}".format
        print(wide.to_string(formatters=fmt))


def print_config_ranking(summary: pd.DataFrame, config_cols: List[str]) -> None:
    print("\n=== Configs ranked by test log-loss (lower is better) ===\n")
    if not config_cols:
        print("(single run — no config columns)")
    cols = (
        config_cols
        + ["test_log_loss_mean", "test_log_loss_std", "test_log_loss_min"]
        + ["test_accuracy_mean", "test_mae_mean", "n_folds", "best_epoch_mean"]
    )
    formatters = {
        "test_log_loss_mean": "{:.4f}".format,
        "test_log_loss_std": "{:.4f}".format,
        "test_log_loss_min": "{:.4f}".format,
        "test_accuracy_mean": "{:.3f}".format,
        "test_mae_mean": "{:.3f}".format,
        "best_epoch_mean": "{:.0f}".format,
    }
    print(
        summary[cols].to_string(
            index=False,
            formatters={k: v for k, v in formatters.items() if k in cols},
        )
    )


def print_feature_importance(fi: pd.DataFrame, config_cols: List[str]) -> None:
    if fi.empty:
        print("\n(no shuffle/ablate feature importance rows)")
        return
    print(
        "\n=== Feature importance (delta = perturbed - baseline test log-loss) ==="
    )
    print("     Higher delta = feature matters more\n")

    if not config_cols:
        for method, sub in fi.groupby("method"):
            print(f"-- method: {method} --")
            print(
                sub.sort_values("delta_mean", ascending=False)[
                    ["feature", "delta_mean", "delta_std", "n_folds"]
                ].to_string(
                    index=False,
                    formatters={
                        "delta_mean": "{:+.4f}".format,
                        "delta_std": "{:.4f}".format,
                    },
                )
            )
            print()
        return

    # Grid case: print each method's ranking grouped by config
    for method in sorted(fi["method"].unique()):
        print(f"=== method: {method} ===")
        sub = fi[fi["method"] == method].copy()
        for config_vals, g in sub.groupby(config_cols, dropna=False):
            if not isinstance(config_vals, tuple):
                config_vals = (config_vals,)
            label = " ".join(
                f"{c}={v}" for c, v in zip(config_cols, config_vals)
            )
            ranked = g.sort_values("delta_mean", ascending=False)[
                ["feature", "delta_mean", "delta_std"]
            ]
            print(f"  [{label}]")
            print(
                ranked.to_string(
                    index=False,
                    formatters={
                        "delta_mean": "{:+.4f}".format,
                        "delta_std": "{:.4f}".format,
                    },
                )
            )
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise AH training metrics from a run or grid search, "
            "with a focus on test log-loss and feature importance."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Artifact dir (containing metrics/), a metrics dir, or a parquet.",
    )
    parser.add_argument(
        "--epoch-mode",
        choices=["best", "last"],
        default="best",
        help="Per-fold epoch selection strategy (default: best test log-loss).",
    )
    parser.add_argument(
        "--mask",
        type=int,
        default=0,
        help="Mask pattern index to analyse (default 0).",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    df = load_metrics(args.path)
    config_cols = detect_config_cols(df)
    print(f"Loaded {len(df):,} rows from {args.path}")
    print(
        "Detected config columns:",
        config_cols if config_cols else "(none — single run)",
    )
    print(f"CV folds: {sorted(df['cv_split'].dropna().unique().tolist())}")
    print(f"Epoch range: {df['epoch'].min()} .. {df['epoch'].max()}")

    # Baseline test log-loss drives epoch selection.
    baseline_ll = df[
        (df["set"] == "test")
        & (df["name"] == "log_loss")
        & (df["mask"] == args.mask)
    ]
    if "shuffle_feature" in baseline_ll.columns:
        baseline_ll = baseline_ll[baseline_ll["shuffle_feature"].isna()]
    if "ablate_feature" in baseline_ll.columns:
        baseline_ll = baseline_ll[baseline_ll["ablate_feature"].isna()]

    if baseline_ll.empty:
        print(
            f"No baseline test log_loss rows at mask={args.mask}. Aborting.",
            file=sys.stderr,
        )
        sys.exit(2)

    chosen = pick_epoch_per_fold(baseline_ll, config_cols, args.epoch_mode)
    summary = test_loss_summary(df, chosen, config_cols, mask=args.mask)
    fi = feature_importance(df, chosen, config_cols, mask=args.mask)

    print_config_ranking(summary, config_cols)
    print_feature_importance(fi, config_cols)
    print_feature_importance_table(fi, summary, config_cols)


if __name__ == "__main__":
    main()
