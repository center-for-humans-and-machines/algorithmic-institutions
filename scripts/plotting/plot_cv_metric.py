"""Plot a cross-validated training metric vs epoch for any number of runs.

Model-agnostic: hand it metrics parquets (or artifact dirs) from any training
run -- contribution, punishment, switch predictor, etc. Each run is drawn as a
curve of the chosen metric averaged across CV folds, with a marker at its best
(min by default) epoch and at its start.

A run arg is either a metrics parquet, or an artifact dir (its `metrics/*.parquet`
is resolved automatically; if a dir holds several, pass the parquet directly).

Usage:
    python scripts/plotting/plot_cv_metric.py <run> [<run> ...] \\
        [--labels L1 L2 ...] [--metric NAME] [--set train|test] \\
        [--best min|max] [--out PATH] [--title T] [--ymin Y] [--ymax Y]

Examples:
    # test log-loss across two runs (defaults)
    python scripts/plotting/plot_cv_metric.py \\
        artifacts/artificial_humans/run_a artifacts/artificial_humans/run_b

    # a different metric / split, with custom labels
    python scripts/plotting/plot_cv_metric.py run_a/metrics/x.parquet run_b \\
        --metric accuracy --set test --best max --labels "A" "B"
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_parquet(run):
    """A run arg is a parquet file or an artifact dir holding metrics/*.parquet."""
    p = Path(run)
    if p.is_file():
        return p
    metrics = p / "metrics" if (p / "metrics").is_dir() else p
    parquets = sorted(metrics.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"no .parquet found under {metrics}")
    if len(parquets) > 1:
        names = "\n  ".join(str(q) for q in parquets)
        raise SystemExit(
            f"{metrics} holds several parquets; pass one explicitly:\n  {names}"
        )
    return parquets[0]


def default_label(path):
    """Readable label from the artifact dir name + epoch/feature tags in the stem."""
    parts = Path(path).parts
    stem = Path(path).stem
    run = parts[-3] if len(parts) >= 3 and parts[-2] == "metrics" else stem
    tags = []
    for tok in stem.split("__"):
        if tok.startswith("epochs_"):
            tags.append(tok.replace("epochs_", "") + "ep")
        elif tok.endswith("_True"):
            tags.append(tok[:-5])
    return f"{run} ({', '.join(tags)})" if tags else run


def fold_mean_curve(path, metric, split):
    """Per-epoch `metric` on `split`, averaged across folds; drops shuffle/ablate rows."""
    df = pd.read_parquet(path)
    m = (df["name"] == metric) & (df["set"] == split)
    for c in ["shuffle_feature", "ablate_feature", "leave_one_in_shuffle_feature"]:
        if c in df.columns:
            m &= df[c].isna()
    b = df[m]
    if b.empty:
        raise SystemExit(
            f"no rows for metric='{metric}' set='{split}' in {path}\n"
            f"  available metrics: {sorted(df['name'].unique())}\n"
            f"  available sets:    {sorted(df['set'].dropna().unique())}"
        )
    curve = b.groupby("epoch")["value"].mean().sort_index()
    return curve.index.values, curve.values, b["cv_split"].nunique()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", help="metrics parquet(s) or artifact dir(s)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="legend labels (one per run); defaults derived from paths")
    ap.add_argument("--metric", default="log_loss", help="metric name (default log_loss)")
    ap.add_argument("--set", dest="split", default="test",
                    help="data split: train or test (default test)")
    ap.add_argument("--best", choices=["min", "max"], default="min",
                    help="whether best = min or max of the metric (default min)")
    ap.add_argument("--report", choices=["best", "last"], default="best",
                    help="legend text value: the best-epoch value or the last-epoch "
                         "value (default best). The best-epoch dot is shown either "
                         "way; use 'last' for short/non-overfit runs still descending.")
    ap.add_argument("--out", default="plots/group_selection/cv_metric.png",
                    help="output image path")
    ap.add_argument("--title", default=None, help="plot title (default derived)")
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        raise SystemExit(f"got {len(args.labels)} labels for {len(args.runs)} runs")

    paths = [resolve_parquet(r) for r in args.runs]
    labels = args.labels or [default_label(p) for p in paths]
    pick = np.argmin if args.best == "min" else np.argmax
    bestfn = np.min if args.best == "min" else np.max

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    los, his = [], []
    for i, (path, lab) in enumerate(zip(paths, labels)):
        ep, mean, n_folds = fold_mean_curve(path, args.metric, args.split)
        best = float(bestfn(mean))
        last = float(mean[-1])
        col = cmap(i % 10)
        # legend reports best or last per --report; the best-epoch dot stays either way
        word, val = ("last", last) if args.report == "last" else ("best", best)
        ax.plot(ep, mean, color=col, linewidth=1.6,
                label=f"{lab}  (start {mean[0]:.3f} -> {word} {val:.4f}, {n_folds} folds)")
        ax.scatter([ep[0]], [mean[0]], color=col, s=40, zorder=3, marker="s",
                   edgecolor="black", linewidth=0.6)
        ax.scatter([ep[int(pick(mean))]], [best], color=col, s=45, zorder=3,
                   edgecolor="black", linewidth=0.6)
        los.append(float(np.min(mean)))
        his.append(float(np.max(mean)))
        print(f"{lab:<44} start {mean[0]:.3f}  best {best:.4f}  last {last:.4f}  ({n_folds} folds)")

    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{args.split} {args.metric} (mean across folds)")
    ax.set_title(args.title or f"Fold-mean {args.split} {args.metric} vs epoch")
    # default: full data range so the starting points are visible; --ymin/--ymax to zoom.
    span = max(his) - min(los)
    lo = args.ymin if args.ymin is not None else min(los) - 0.03 * span
    hi = args.ymax if args.ymax is not None else max(his) + 0.03 * span
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
