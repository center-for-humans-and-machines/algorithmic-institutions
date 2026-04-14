#!/usr/bin/env python3
"""Quick CLI summary for punishment training metrics parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_metric(df: pd.DataFrame, set_name: str, metric_name: str) -> None:
    sub = df[(df["set"] == set_name) & (df["name"] == metric_name) & (df["mask"] == 0)]
    if sub.empty:
        print(f"\n[{set_name}] {metric_name}: no rows found for mask=0")
        return

    sub = sub.sort_values(["cv_split", "epoch"])
    first = sub.groupby("cv_split", as_index=False).head(1)
    last = sub.groupby("cv_split", as_index=False).tail(1)
    best = sub.loc[sub.groupby("cv_split")["value"].idxmin()].sort_values("cv_split")

    print(f"\n[{set_name}] {metric_name} (mask=0)")
    print(
        f"  first mean: {first['value'].mean():.6f} | "
        f"last mean: {last['value'].mean():.6f}"
    )
    print(
        f"  last std: {last['value'].std(ddof=0):.6f} | "
        f"best mean: {best['value'].mean():.6f}"
    )
    print("  final by fold:")
    print(last[["cv_split", "epoch", "value"]].to_string(index=False))
    print("  best by fold:")
    print(best[["cv_split", "epoch", "value"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze punishment training metrics parquet."
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Either metrics parquet path or artifact dir "
            "(e.g. artifacts/.../punishment_autoregressive/)."
        ),
    )
    args = parser.parse_args()

    input_path = args.path
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    if input_path.is_dir():
        metrics_dir = input_path / "metrics"
        if not metrics_dir.exists():
            raise FileNotFoundError(
                f"Metrics dir not found under artifact dir: {metrics_dir}"
            )
        candidates = sorted(metrics_dir.glob("*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No parquet file found in: {metrics_dir}")
        if len(candidates) > 1:
            print(
                "Multiple metrics parquet files found; "
                f"using the first one: {candidates[0]}"
            )
        metrics_path = candidates[0]
    else:
        metrics_path = input_path

    df = pd.read_parquet(metrics_path)
    print(f"Loaded: {metrics_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    if "set" in df.columns:
        print(f"Sets: {sorted(df['set'].dropna().unique().tolist())}")
    if "name" in df.columns:
        print(f"Metrics: {sorted(df['name'].dropna().unique().tolist())}")
    if "cv_split" in df.columns:
        print(f"CV splits: {sorted(df['cv_split'].dropna().unique().tolist())}")

    summarize_metric(df, "train", "log_loss")
    summarize_metric(df, "test", "log_loss")
    summarize_metric(df, "train", "accuracy")
    summarize_metric(df, "test", "accuracy")


if __name__ == "__main__":
    main()
