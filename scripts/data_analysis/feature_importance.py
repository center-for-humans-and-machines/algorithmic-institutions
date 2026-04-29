"""Model-level feature importance from shuffle/ablation metrics.

Usage:
    python scripts/data_analysis/feature_importance.py <metrics_parquet>
    python scripts/data_analysis/feature_importance.py <metrics_parquet> \
        --metric log_loss --method shuffle --save-fig out.png

Reads a metrics parquet file produced by the AH training pipeline
and computes feature importance as the change in a metric when a
feature is shuffled or ablated, compared to the unperturbed baseline.

Importance = metric_perturbed - metric_baseline  (for loss/error metrics)

Reports the final-epoch, test-set, CV-averaged importance per feature.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PERTURB_COLS = [
    "shuffle_feature",
    "ablate_feature",
    "leave_one_in_shuffle_feature",
    "leave_one_in_ablate_feature",
]


def load_metrics(path):
    df = pd.read_parquet(path)
    available = {
        col: col in df.columns and df[col].notna().any()
        for col in PERTURB_COLS
    }
    if not any(available.values()):
        raise ValueError(
            "No feature importance data found in "
            f"{path}"
        )
    return df, available


def compute_total_improvement(df, metric, strategy):
    """Compute avg test loss improvement from epoch 0 to final."""
    final_epoch = int(df["epoch"].max())

    filt = (df["set"] == "test") & (df["name"] == metric)
    if strategy is not None:
        filt = filt & (df["strategy"] == strategy)

    # Exclude perturbed rows
    for col in PERTURB_COLS:
        if col in df.columns:
            filt = filt & df[col].isna()

    sub = df[filt]
    epoch0 = sub[sub["epoch"] == 0].groupby("cv_split")["value"].mean()
    final = sub[sub["epoch"] == final_epoch].groupby("cv_split")[
        "value"
    ].mean()
    improvement = epoch0 - final
    return improvement.mean(), improvement.std(), epoch0.mean(), final.mean()


def compute_importance(df, method_col, metric, strategy):
    """Compute per-feature importance as delta from baseline.

    Returns a DataFrame with columns:
        feature, baseline, perturbed, delta, cv_split
    averaged across CV splits at the final epoch, on test set.
    """
    final_epoch = int(df["epoch"].max())

    # Filter to test set, final epoch, chosen metric/strategy
    filt = (
        (df["set"] == "test")
        & (df["epoch"] == final_epoch)
        & (df["name"] == metric)
    )
    if strategy is not None:
        filt = filt & (df["strategy"] == strategy)
    sub = df[filt].copy()

    # Baseline: rows where ALL perturbation columns are null
    base_filt = pd.Series(True, index=sub.index)
    for col in PERTURB_COLS:
        if col in sub.columns:
            base_filt = base_filt & sub[col].isna()
    baseline = sub[base_filt]
    # Perturbed: rows where the method column is set
    perturbed = sub[sub[method_col].notna()]

    if baseline.empty:
        raise ValueError(
            f"No baseline rows for metric={metric}, "
            f"strategy={strategy}, epoch={final_epoch}"
        )
    if perturbed.empty:
        raise ValueError(
            f"No {method_col} rows for metric={metric}, "
            f"strategy={strategy}, epoch={final_epoch}"
        )

    # Average baseline per CV split
    base_avg = (
        baseline.groupby("cv_split")["value"]
        .mean()
        .rename("baseline")
    )

    # Average perturbed per (cv_split, feature)
    pert_avg = (
        perturbed.groupby(["cv_split", method_col])["value"]
        .mean()
        .rename("perturbed")
        .reset_index()
        .rename(columns={method_col: "feature"})
    )

    pert_avg = pert_avg.join(base_avg, on="cv_split")
    pert_avg["delta"] = pert_avg["perturbed"] - pert_avg["baseline"]

    return pert_avg


def print_importance(imp_df, method_label, metric):
    """Print a ranked importance table."""
    summary = (
        imp_df.groupby("feature")
        .agg(
            baseline=("baseline", "mean"),
            perturbed=("perturbed", "mean"),
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
        )
        .sort_values("delta_mean", ascending=False)
    )

    print(f"=== {method_label} Feature Importance ({metric}) ===")
    print(
        f"  {'feature':<30s} {'baseline':>10s} "
        f"{'perturbed':>10s} {'delta':>10s} {'std':>8s}"
    )
    for feat, row in summary.iterrows():
        bar = "#" * max(0, int(row["delta_mean"] * 40))
        print(
            f"  {feat:<30s} {row['baseline']:>10.4f} "
            f"{row['perturbed']:>10.4f} "
            f"{row['delta_mean']:>+10.4f} "
            f"{row['delta_std']:>8.4f}  {bar}"
        )
    print()
    return summary


def plot_importance(rows, metric, save_path, total_improvement=None):
    """Plot horizontal bar charts of feature importance.

    Parameters
    ----------
    rows : list of list of (label, summary_df) tuples
        Each inner list is one row of panels.
    metric : str
    save_path : str or Path
    total_improvement : tuple (mean, std, loss_0, loss_final) or None
    """
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    n_feats = max(
        len(s) for row in rows for _, s in row
    )
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, max(3, 0.5 * n_feats) * n_rows),
        squeeze=False,
    )

    for r, row in enumerate(rows):
        for c, (label, summary) in enumerate(row):
            ax = axes[r][c]
            summary = summary.sort_values("delta_mean")
            ax.barh(
                summary.index,
                summary["delta_mean"],
                xerr=summary["delta_std"],
                capsize=3,
                color="#4c72b0",
                edgecolor="white",
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel(f"\u0394 {metric}")
            ax.set_title(f"{label} importance")

            if total_improvement is not None:
                imp_mean = total_improvement[0]
                ax.axvline(
                    imp_mean,
                    color="#c44e52",
                    linewidth=1.5,
                    linestyle="--",
                    label=f"total gain ({imp_mean:.3f})",
                )
                ax.legend(fontsize=8, loc="lower right")

        # Hide unused axes in this row
        for c in range(len(row), n_cols):
            axes[r][c].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Model-level feature importance from "
        "shuffle/ablation metrics"
    )
    parser.add_argument(
        "metrics",
        help="Path to metrics parquet file",
    )
    parser.add_argument(
        "--metric",
        default="log_loss",
        help="Metric name to use (default: log_loss)",
    )
    parser.add_argument(
        "--method",
        choices=["shuffle", "ablate", "both"],
        default="both",
        help="Which perturbation method to report "
        "(default: both)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Prediction strategy filter "
        "(default: None = use log_loss rows)",
    )
    parser.add_argument(
        "--save-fig",
        default=None,
        metavar="PATH",
        help="Save a bar-chart figure to PATH",
    )
    args = parser.parse_args()

    df, available = load_metrics(args.metrics)

    final_epoch = int(df["epoch"].max())
    n_splits = int(df["cv_split"].dropna().nunique())

    imp_mean, imp_std, loss_0, loss_final = compute_total_improvement(
        df, args.metric, args.strategy
    )

    print(f"Metrics: {args.metrics}")
    print(f"Final epoch: {final_epoch}, CV splits: {n_splits}")
    print(f"Metric: {args.metric}, Strategy: {args.strategy}")
    for col, present in available.items():
        if present:
            feats = sorted(df[col].dropna().unique())
            print(f"{col}: {feats}")
    print()
    print(
        f"Total test {args.metric} improvement: "
        f"{loss_0:.4f} -> {loss_final:.4f} "
        f"(delta={imp_mean:+.4f} +/- {imp_std:.4f})"
    )
    print()

    # Map (column, label) for each perturbation method
    method_map = {
        "shuffle": [
            ("shuffle_feature", "Shuffle"),
            ("leave_one_in_shuffle_feature", "Leave-one-in (shuffle)"),
        ],
        "ablate": [
            ("ablate_feature", "Ablation"),
            ("leave_one_in_ablate_feature", "Leave-one-in (ablation)"),
        ],
    }

    rows = []
    methods = (
        ["shuffle", "ablate"]
        if args.method == "both"
        else [args.method]
    )
    for method in methods:
        row = []
        for col, label in method_map[method]:
            if available.get(col):
                imp = compute_importance(
                    df, col, args.metric, args.strategy
                )
                summary = print_importance(
                    imp, label, args.metric
                )
                row.append((label, summary))
        if row:
            rows.append(row)

    if args.save_fig and rows:
        save_path = Path(args.save_fig)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plot_importance(
            rows,
            args.metric,
            save_path,
            total_improvement=(imp_mean, imp_std, loss_0, loss_final),
        )


if __name__ == "__main__":
    main()
