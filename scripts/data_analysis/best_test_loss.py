"""Compare models by best test log-loss across CV folds.

For each metrics parquet in the given artifact dir(s), computes:
- per-fold best test log_loss
- mean ± std across folds
- globally-best epoch (the epoch where the fold-averaged test loss is min)
- per-fold best epochs

Prints a table sorted by mean best test loss.

Usage:
    python scripts/data_analysis/best_test_loss.py <artifact_dir> [<artifact_dir> ...]
    python scripts/data_analysis/best_test_loss.py artifacts/artificial_humans/switch_pred_opt
    python scripts/data_analysis/best_test_loss.py \
        artifacts/artificial_humans/switch_pred_grid_search_35ep \
        artifacts/artificial_humans/switch_pred_opt
"""

import argparse
from pathlib import Path

import pandas as pd


def analyze_parquet(path):
    df = pd.read_parquet(path)
    mask = (df["name"] == "log_loss") & (df["set"] == "test")
    if "shuffle_feature" in df.columns:
        mask &= df["shuffle_feature"].isna()
    if "ablate_feature" in df.columns:
        mask &= df["ablate_feature"].isna()
    base = df[mask]
    if base.empty:
        return None

    per_fold_best = base.groupby("cv_split")["value"].min()
    idx = base.groupby("cv_split")["value"].idxmin()
    per_fold = base.loc[idx, ["cv_split", "epoch", "value"]].sort_values("cv_split")

    mean_ep = base.groupby("epoch")["value"].mean()
    globally_best_ep = int(mean_ep.idxmin())
    globally_best_loss = float(mean_ep.loc[globally_best_ep])
    final_ep = int(base["epoch"].max())
    final_loss = float(mean_ep.loc[final_ep])

    return {
        "best_mean": float(per_fold_best.mean()),
        "best_std": float(per_fold_best.std()),
        "per_fold_best": [round(float(v), 4) for v in per_fold_best.values],
        "best_ep_per_fold": per_fold["epoch"].astype(int).tolist(),
        "globally_best_ep": globally_best_ep,
        "globally_best_loss": globally_best_loss,
        "final_ep": final_ep,
        "final_loss": final_loss,
    }


def collect(artifact_dirs):
    rows = []
    for d in artifact_dirs:
        d = Path(d)
        metrics_dir = d / "metrics"
        if not metrics_dir.exists():
            print(f"[warn] no metrics dir: {metrics_dir}")
            continue
        for f in sorted(metrics_dir.glob("*.parquet")):
            stats = analyze_parquet(f)
            if stats is None:
                continue
            rows.append({
                "artifact": d.name,
                "job": f.stem,
                **stats,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Rank AH runs by mean best test log-loss across folds."
    )
    parser.add_argument(
        "artifact_dirs",
        nargs="+",
        help="One or more artifact dirs (each containing metrics/*.parquet).",
    )
    parser.add_argument(
        "--show-epochs",
        action="store_true",
        help="Also show per-fold best epochs.",
    )
    args = parser.parse_args()

    df = collect(args.artifact_dirs)
    if df.empty:
        print("No metrics found.")
        return

    df = df.sort_values("best_mean").reset_index(drop=True)

    cols = ["artifact", "job", "best_mean", "best_std",
            "globally_best_ep", "final_loss"]
    if args.show_epochs:
        cols.append("best_ep_per_fold")
        cols.append("per_fold_best")

    display = df[cols].copy()
    display["best_mean"] = display["best_mean"].round(4)
    display["best_std"] = display["best_std"].round(4)
    display["final_loss"] = display["final_loss"].round(4)

    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
