"""Plot confusion matrix from AH training artifacts.

Usage:
    python scripts/plot_confusion_matrix.py <artifact_dir> [--output <path>]

Example:
    python scripts/plot_confusion_matrix.py \
        artifacts/artificial_humans/switch_predictor

    python scripts/plot_confusion_matrix.py \
        artifacts/artificial_humans/switch_predictor \
        --output plots/group_selection/confusion_matrix.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def load_predictions(artifact_dir, job_id=None):
    cm_dir = os.path.join(artifact_dir, "confusion_matrix")
    files = [f for f in os.listdir(cm_dir) if f.endswith(".parquet")]
    if job_id is not None:
        files = [f for f in files if f == f"{job_id}.parquet"]
    if not files:
        print(f"No parquet files in {cm_dir}", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(
        [pd.read_parquet(os.path.join(cm_dir, f)) for f in files]
    )

    # Keep only valid (decision-round) predictions
    valid = df[df["valid"] == True].copy()  # noqa: E712

    # Determine target column name (e.g. true_does_switch, true_contribution)
    true_cols = [c for c in valid.columns if c.startswith("true_")]
    if len(true_cols) != 1:
        print(
            f"Expected 1 true_* column, found: {true_cols}",
            file=sys.stderr,
        )
        sys.exit(1)
    true_col = true_cols[0]
    target_name = true_col.replace("true_", "")
    pred_col = f"pred_{target_name}"

    # Each sample has n_levels rows (one per class probability).
    # Pick the class with highest probability as the prediction.
    group_cols = ["idx", "round_number"]
    if (
        "cv_split" in valid.columns
        and valid["cv_split"].notna().any()
    ):
        group_cols.append("cv_split")

    def _agg(g):
        return pd.Series(
            {
                "true": int(g[true_col].iloc[0]),
                "pred": int(
                    g.loc[g["proba"].idxmax(), pred_col]
                ),
            }
        )

    pred = (
        valid.groupby(group_cols)[[true_col, pred_col, "proba"]]
        .apply(_agg)
        .reset_index()
    )
    return pred, target_name


def plot(pred, target_name, output_path):
    y_true = pred["true"].values
    y_pred = pred["pred"].values
    labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Annotation: count + percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.0%})"

    if target_name == "does_switch":
        tick_labels = ["stay", "switch"]
    else:
        tick_labels = [str(l) for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {target_name}")

    # Print classification report to stdout
    report = classification_report(
        y_true, y_pred, target_names=tick_labels, zero_division=0
    )
    print(report)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot confusion matrix from AH artifacts"
    )
    parser.add_argument(
        "artifact_dir",
        help="Path to artifact directory (e.g. artifacts/artificial_humans/switch_predictor)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output plot path (default: plots/group_selection/<model_name>.png)",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help=(
            "Filter to a single grid-search job id "
            "(matches <job_id>.parquet under confusion_matrix/)."
        ),
    )
    args = parser.parse_args()

    pred, target_name = load_predictions(args.artifact_dir, job_id=args.job_id)

    if args.output:
        output_path = args.output
    else:
        base_name = os.path.basename(args.artifact_dir.rstrip("/"))
        model_name = f"{base_name}__{args.job_id}" if args.job_id else base_name
        output_path = os.path.join(
            "plots", "group_selection", f"{model_name}.png"
        )

    plot(pred, target_name, output_path)


if __name__ == "__main__":
    main()
