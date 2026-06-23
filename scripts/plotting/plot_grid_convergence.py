"""Test log-loss convergence curves from training metrics parquets.

Reads the metrics parquet(s) a training run writes -- a single-run
artifact dir, its metrics/ subfolder, or a grid dir with one parquet per
run -- and plots one mean-across-folds test log-loss curve per run, over
epochs. Two panels: the full schedule and a zoom on the converged region.
Best-epoch points are marked.

Baseline (unshuffled) test log-loss is the row where ALL perturbation
columns are null: shuffle_feature, leave_one_in_shuffle_feature,
ablate_feature. The trainer also records shuffle / leave-one-in / ablate
perturbation rows (for feature importance); those carry a non-null value
in one of those columns. Filtering on shuffle_feature alone is NOT enough
-- leave-one-in rows have shuffle_feature=null too -- so all three must
be excluded to isolate the true baseline.

Each run is coloured by base architecture (the label with any `_ag`
suffix stripped); agent_group variants are dashed.

Usage:
    python scripts/plotting/plot_grid_convergence.py <artifact_dir> \\
        [--title "Contribution"] [--zoom-x 300 2000] [--zoom-y 1.95 2.15] \\
        [--out FILE.jpg]
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

PERTURB_COLS = ("shuffle_feature", "leave_one_in_shuffle_feature", "ablate_feature")


def find_parquets(path: str) -> list:
    if os.path.isfile(path):
        return [path]
    metrics_dir = os.path.join(path, "metrics")
    metrics_dir = metrics_dir if os.path.isdir(metrics_dir) else path
    return sorted(glob.glob(os.path.join(metrics_dir, "*.parquet")))


def baseline_curve(df: pd.DataFrame) -> pd.Series:
    """Mean-across-folds baseline test log-loss, indexed by epoch.

    Baseline = all perturbation columns null (excludes shuffle /
    leave-one-in / ablate rows).
    """
    m = df[(df["name"] == "log_loss") & (df["set"] == "test")]
    for col in PERTURB_COLS:
        if col in m.columns:
            m = m[m[col].isna()]
    return m.pivot_table(index="epoch", columns="cv_split", values="value").mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact_dir",
        help="Run artifact dir, its metrics/ dir, or a single parquet",
    )
    parser.add_argument("--title", default="AH grid",
                        help="Figure title prefix, e.g. 'Contribution'")
    parser.add_argument("--zoom-x", type=float, nargs=2, default=[300, 2000])
    parser.add_argument("--zoom-y", type=float, nargs=2, default=[1.95, 2.15])
    parser.add_argument(
        "--out", default="plots/group_selection/grid_convergence.jpg",
        help="Output figure path",
    )
    args = parser.parse_args()

    files = find_parquets(args.artifact_dir)
    if not files:
        sys.exit(f"No metrics parquets found under {args.artifact_dir}")

    curves = {}
    n_folds = 0
    for f in files:
        df = pd.read_parquet(f)
        label = str(df["architecture"].iloc[0])
        n_folds = max(n_folds, df["cv_split"].nunique())
        curves[label] = baseline_curve(df)

    # colour by base architecture (strip _ag); _ag drawn dashed
    bases = sorted({lbl[:-3] if lbl.endswith("_ag") else lbl for lbl in curves})
    cmap = plt.get_cmap("tab10")
    base_color = {b: cmap(i % 10) for i, b in enumerate(bases)}

    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
    )
    for label in sorted(curves, key=lambda lbl: curves[lbl].min()):
        c = curves[label]
        is_ag = label.endswith("_ag")
        color = base_color[label[:-3] if is_ag else label]
        style = "--" if is_ag else "-"
        best = c.idxmin()
        for a in (ax, axz):
            a.plot(c.index, c.values, color=color, ls=style, lw=1.8, label=label)
            a.scatter([best], [c.loc[best]], color=color, s=35,
                      edgecolor="white", zorder=5)

    ax.set_xlabel("epoch")
    ax.set_ylabel(f"test log-loss (mean across {n_folds} folds)")
    ax.set_title("Full training curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=1, loc="upper right")

    axz.set_xlim(*args.zoom_x)
    axz.set_ylim(*args.zoom_y)
    axz.set_xlabel("epoch")
    axz.set_title("Zoom: converged region")
    axz.grid(alpha=0.3)

    fig.suptitle(
        f"{args.title} — test log-loss convergence "
        "(dashed = agent_group variants)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
