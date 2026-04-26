"""Plot cross-validation summary from AH training artifacts.

Produces:
1. Loss curve plot (train + test, mean +/- std across folds)
2. Per-class precision/recall/F1 table (mean +/- std across folds)

Usage:
    python scripts/plotting/plot_cv_summary.py <artifact_dir>
    python scripts/plotting/plot_cv_summary.py <artifact_dir> \
        --output-dir plots/group_selection

Example:
    python scripts/plotting/plot_cv_summary.py \
        artifacts/artificial_humans/switch_pred_mlp_test
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metrics(artifact_dir, job_id=None):
    metrics_dir = os.path.join(artifact_dir, "metrics")
    files = [f for f in os.listdir(metrics_dir) if f.endswith(".parquet")]
    if job_id is not None:
        files = [f for f in files if f == f"{job_id}.parquet"]
    if not files:
        print(f"No parquet files in {metrics_dir}", file=sys.stderr)
        sys.exit(1)
    return pd.concat(
        [pd.read_parquet(os.path.join(metrics_dir, f)) for f in files]
    )


def load_confusion_matrix(artifact_dir, job_id=None):
    cm_dir = os.path.join(artifact_dir, "confusion_matrix")
    files = [f for f in os.listdir(cm_dir) if f.endswith(".parquet")]
    if job_id is not None:
        files = [f for f in files if f == f"{job_id}.parquet"]
    if not files:
        print(f"No parquet files in {cm_dir}", file=sys.stderr)
        sys.exit(1)
    return pd.concat(
        [pd.read_parquet(os.path.join(cm_dir, f)) for f in files]
    )


def _extract_loss_curves(metrics):
    """Return (train_pivot, test_pivot) with epoch index, cv_split cols."""
    has_shuffle = "shuffle_feature" in metrics.columns
    train = metrics[
        (metrics["set"] == "train")
        & (metrics["name"] == "loss")
        & (metrics["strategy"].isna())
    ]
    test = metrics[
        (metrics["set"] == "test")
        & (metrics["name"] == "log_loss")
        & (metrics["strategy"].isna())
    ]
    if has_shuffle:
        train = train[train["shuffle_feature"].isna()]
        test = test[test["shuffle_feature"].isna()]
    train_pivot = train.pivot_table(
        index="epoch", columns="cv_split", values="value"
    )
    test_pivot = test.pivot_table(
        index="epoch", columns="cv_split", values="value"
    )
    return train_pivot, test_pivot


def _plot_series(ax, pivot, color, label, linestyle="-"):
    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1)
    for col in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[col],
            color=color,
            alpha=0.15,
            linewidth=0.8,
            linestyle=linestyle,
        )
    ax.plot(
        mean.index,
        mean.values,
        color=color,
        label=label,
        linewidth=2,
        linestyle=linestyle,
    )
    ax.fill_between(
        mean.index,
        (mean - std).values,
        (mean + std).values,
        color=color,
        alpha=0.2,
    )


def plot_loss_curves(metrics, output_path, title=None, compare=None):
    """Plot train loss and test log loss across CV folds.

    If `compare` is given as (metrics_df, label_primary, label_compare),
    overlays a second model's curves with dashed lines.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    train_pivot, test_pivot = _extract_loss_curves(metrics)

    if compare is None:
        _plot_series(ax, train_pivot, "tab:blue", "Train loss")
        _plot_series(ax, test_pivot, "tab:orange", "Test log loss")
    else:
        compare_metrics, label_primary, label_compare = compare
        c_train, c_test = _extract_loss_curves(compare_metrics)
        _plot_series(
            ax, train_pivot, "tab:blue", f"Train ({label_primary})"
        )
        _plot_series(
            ax, test_pivot, "tab:orange", f"Test ({label_primary})"
        )
        _plot_series(
            ax,
            c_train,
            "tab:green",
            f"Train ({label_compare})",
            linestyle="--",
        )
        _plot_series(
            ax,
            c_test,
            "tab:red",
            f"Test ({label_compare})",
            linestyle="--",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    if title:
        ax.set_title(title)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved loss plot to {output_path}")
    plt.close(fig)


def compute_class_metrics(cm_df):
    """Compute per-class precision/recall/F1 per CV fold."""
    valid = cm_df[cm_df["valid"] == True].copy()  # noqa: E712

    true_cols = [c for c in valid.columns if c.startswith("true_")]
    true_col = true_cols[0]
    target_name = true_col.replace("true_", "")
    pred_col = f"pred_{target_name}"

    # Resolve greedy predictions (argmax proba per sample)
    group_cols = ["idx", "round_number"]
    has_cv = (
        "cv_split" in valid.columns
        and valid["cv_split"].notna().any()
    )
    if has_cv:
        group_cols.append("cv_split")

    def _agg(g):
        return pd.Series(
            {
                "true": int(g[true_col].iloc[0]),
                "pred": int(
                    g.loc[g["proba"].idxmax(), pred_col]
                ),
                "cv_split": (
                    g["cv_split"].iloc[0] if has_cv else 0
                ),
            }
        )

    pred = (
        valid.groupby(group_cols)[[
            true_col, pred_col, "proba", "cv_split"
        ]]
        .apply(_agg)
        .reset_index(drop=True)
    )

    if target_name == "does_switch":
        class_names = {0: "stay", 1: "switch"}
    else:
        class_names = {
            v: str(v)
            for v in sorted(pred["true"].unique())
        }

    rows = []
    for fold in sorted(pred["cv_split"].unique()):
        fold_data = pred[pred["cv_split"] == fold]
        y_true = fold_data["true"].values
        y_pred = fold_data["pred"].values

        for cls_id, cls_name in class_names.items():
            tp = ((y_pred == cls_id) & (y_true == cls_id)).sum()
            fp = ((y_pred == cls_id) & (y_true != cls_id)).sum()
            fn = ((y_pred != cls_id) & (y_true == cls_id)).sum()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * p * r / (p + r) if (p + r) > 0 else 0.0
            )

            rows.append(
                {
                    "fold": int(fold),
                    "class": cls_name,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "support": int((y_true == cls_id).sum()),
                }
            )

    return pd.DataFrame(rows)


def print_summary(class_metrics):
    """Print mean +/- std of per-class metrics across folds."""
    summary = (
        class_metrics.groupby("class")[
            ["precision", "recall", "f1", "support"]
        ]
        .agg(["mean", "std"])
    )

    print("\n=== Per-class metrics (mean +/- std across folds) ===\n")
    header = (
        f"{'Class':<10s} {'Precision':>14s} {'Recall':>14s}"
        f" {'F1':>14s} {'Support':>10s}"
    )
    print(header)
    print("-" * len(header))

    macro_f1s = []
    for cls in summary.index:
        p_m, p_s = (
            summary.loc[cls, ("precision", "mean")],
            summary.loc[cls, ("precision", "std")],
        )
        r_m, r_s = (
            summary.loc[cls, ("recall", "mean")],
            summary.loc[cls, ("recall", "std")],
        )
        f_m, f_s = (
            summary.loc[cls, ("f1", "mean")],
            summary.loc[cls, ("f1", "std")],
        )
        sup = summary.loc[cls, ("support", "mean")]
        print(
            f"{cls:<10s}"
            f" {p_m:5.1%} +/- {p_s:5.1%}"
            f" {r_m:5.1%} +/- {r_s:5.1%}"
            f" {f_m:5.1%} +/- {f_s:5.1%}"
            f" {sup:10.0f}"
        )
        macro_f1s.append(f_m)

    macro = np.mean(macro_f1s)
    print(f"\n{'Macro F1:':<10s} {macro:5.1%}")

    # Per-fold detail
    print("\n=== Per-fold detail ===\n")
    for fold in sorted(class_metrics["fold"].unique()):
        fold_data = class_metrics[class_metrics["fold"] == fold]
        parts = []
        for _, row in fold_data.iterrows():
            parts.append(
                f"{row['class']}: P={row['precision']:.1%}"
                f" R={row['recall']:.1%}"
                f" F1={row['f1']:.1%}"
            )
        print(f"  Fold {fold}: {' | '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot CV summary from AH artifacts"
    )
    parser.add_argument(
        "artifact_dir",
        help="Path to artifact directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for plots "
            "(default: plots/group_selection)"
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Title for loss curve plot",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help=(
            "Filter to a single grid-search job id "
            "(matches <job_id>.parquet under metrics/ and confusion_matrix/)."
        ),
    )
    parser.add_argument(
        "--compare",
        default=None,
        help=(
            "Second artifact directory to overlay on the loss plot "
            "(dashed lines). Skips per-class summary."
        ),
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.artifact_dir.rstrip("/"))
    model_name = f"{base_name}__{args.job_id}" if args.job_id else base_name
    output_dir = args.output_dir or os.path.join(
        "plots", "group_selection"
    )

    metrics = load_metrics(args.artifact_dir, job_id=args.job_id)

    if args.compare:
        compare_name = os.path.basename(args.compare.rstrip("/"))
        compare_metrics = load_metrics(args.compare)
        loss_path = os.path.join(
            output_dir,
            f"{model_name}_vs_{compare_name}_loss_cv.png",
        )
        plot_loss_curves(
            metrics,
            loss_path,
            title=args.title or f"{base_name} vs {compare_name}",
            compare=(compare_metrics, base_name, compare_name),
        )
        return

    # Loss curves
    loss_path = os.path.join(
        output_dir, f"{model_name}_loss_cv.png"
    )
    plot_loss_curves(
        metrics, loss_path, title=args.title or model_name
    )

    # Per-class metrics
    cm_df = load_confusion_matrix(args.artifact_dir, job_id=args.job_id)
    class_metrics = compute_class_metrics(cm_df)
    print_summary(class_metrics)


if __name__ == "__main__":
    main()
