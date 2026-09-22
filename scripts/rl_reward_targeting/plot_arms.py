"""Two figures for the reward-targeting comparison.

`policy_shape_arms.jpg` -- mean punishment against contribution bin, the two
arms drawn seed against paired seed, with the human and clone references. If
the reward were driving the inversion the per-capita curves would sit on the
human side of their pool twins.

`paired_differences.jpg` -- the three paired differences per statistic, drawn
individually rather than as a mean, against the measured sim-seed noise floor.
Three seeds is only informative through the pairing, so the three points are
the result and their average is not.

Usage:
    python scripts/rl_reward_targeting/plot_arms.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure_arms import OUT, PERCAPITA, POOL, SEEDS  # noqa: E402

POOL_C = "#B4553F"
PC_C = "#2F6F8F"
HUMAN_C = "#1A1A1A"
CLONE_C = "#8A8A8A"


def shape_figure(shape, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, seed, a, b in zip(axes, SEEDS, POOL, PERCAPITA):
        x = range(len(shape.index))
        ax.plot(
            x, shape["human managers"], color=HUMAN_C, lw=2.4, label="human", zorder=5
        )
        if "lin_punisher" in shape:
            ax.plot(
                x,
                shape["lin_punisher"],
                color=CLONE_C,
                lw=2.0,
                ls=":",
                label="clone",
                zorder=4,
            )
        if a in shape:
            ax.plot(x, shape[a], color=POOL_C, lw=2.2, marker="o", label="pool")
        if b in shape:
            ax.plot(
                x,
                shape[b],
                color=PC_C,
                lw=2.2,
                marker="s",
                ls="--",
                label="per-capita",
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(shape.index, rotation=45, ha="right")
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("contribution")
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("mean punishment")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Policy shape, paired by seed: the human curve falls, an inverted "
        "manager rises",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def paired_figure(pair, floor, path, cols):
    fig, axes = plt.subplots(1, len(cols), figsize=(3.1 * len(cols), 4.0))
    for ax, c in zip(axes, cols):
        d = pair[f"d_{c}"]
        ax.axhline(0, color="#555", lw=1.0)
        if floor is not None and c in floor.columns:
            learned = [m for m in POOL + PERCAPITA if m in floor.index]
            f = float(floor.loc[learned, c].mean())
            ax.axhspan(-f, f, color="#CCCCCC", alpha=0.55, lw=0, label="noise floor")
        ax.scatter(
            range(len(d)), d.values, s=110, color=PC_C, zorder=5, label="per seed"
        )
        for i, (s, v) in enumerate(d.items()):
            ax.annotate(
                f"{v:+.2f}",
                (i, v),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
            )
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([f"s{s}" for s in d.index])
        ax.set_title(c, fontsize=10)
        ax.grid(alpha=0.25, lw=0.6, axis="y")
    axes[0].set_ylabel("per-capita minus pool")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "The three paired differences, individually. Inside the grey band is "
        "no difference at all.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    shape = pd.read_csv(os.path.join(OUT, "policy_shape.csv"), index_col=0)
    pair = pd.read_csv(os.path.join(OUT, "paired_differences.csv"), index_col=0)
    floor_path = os.path.join(OUT, "noise_floor.csv")
    floor = pd.read_csv(floor_path, index_col=0) if os.path.exists(floor_path) else None
    shape_figure(shape, os.path.join(OUT, "policy_shape_arms.jpg"))
    paired_figure(
        pair,
        floor,
        os.path.join(OUT, "paired_differences.jpg"),
        ["targeting_rho", "shape_delta", "leaver_gap", "mean_punishment"],
    )
    print("wrote", os.path.join(OUT, "policy_shape_arms.jpg"))
    print("wrote", os.path.join(OUT, "paired_differences.jpg"))


if __name__ == "__main__":
    main()
