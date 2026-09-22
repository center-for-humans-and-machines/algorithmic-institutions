"""Tables and figures for the sigmoid-rule arm.

Everything here is read off two sweeps: the design sweep (fit seeds) and the
validation sweep (held-out seeds). Both objectives are carried separately
everywhere -- there is no combined score in this file on purpose.

Usage:
    PYTHONPATH=src python scripts/rule_sigmoid/report.py \
        --design-run runs/sweep_clone --design runs/design_1024.csv \
        --valid-run runs/valid_clone --valid-design runs/validation.csv \
        --fit-dir plots/data_analysis/rule_sigmoid --fit-seeds 42,43 \
        --holdout-seeds 44,45,46 --out plots/data_analysis/rule_sigmoid
"""

import argparse
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from aggregate import (  # noqa: E402
    OBJECTIVES,
    load_sweep,
    paired_bootstrap,
    per_point,
)
from aimanager.evaluation_suite.convert import load_human  # noqa: E402
from aimanager.manager.paired_rollout import RPA_EDGES, RPA_LABELS  # noqa: E402
from fit_surrogate import AXES, HI, LO  # noqa: E402

#: The rows a reader compares everything against.
REFERENCES = ("thr9_p10", "never", "ah_punisher")

#: Rules whose policy shape is drawn beside the human and the clone.
SHAPE_ROWS = ("opt_contribution", "opt_pool", "thr9_p10", "ah_punisher", "never")


def human_policy_shape(csv_path):
    """Mean human punishment per contribution bin, the evaluation suite's
    own convention: one copy per game, rows where both the player and the
    manager gave an input."""
    df = load_human(csv_path)
    v = df.dropna(subset=["punishment", "contribution"])
    bins = pd.cut(v["contribution"], list(RPA_EDGES), labels=list(RPA_LABELS))
    out = v.groupby(bins.astype(str), observed=False)["punishment"].mean()
    return out.reindex(list(RPA_LABELS))


def validation_table(episodes, design, seeds):
    t = design.merge(per_point(episodes, seeds=seeds), on="name", how="inner")
    cols = [
        "name",
        "kind",
        "p_max",
        "c0",
        "tau",
        "gamma_ep",
        "gamma_sw",
        "focal_contribution",
        "focal_pool",
        "focal_members",
        "mean_p",
        "mean_c_valid",
        "c_gap",
        "p_gap",
        "leave_rate",
        "rival_pool",
        "rival_members",
        "n_episodes",
    ]
    return t[cols].sort_values("focal_pool", ascending=False).reset_index(drop=True)


def contrasts(episodes, design, seeds):
    df = episodes[episodes["seed"].isin(seeds)]
    names = [n for n in design["name"] if n in set(df["name"])]
    rows = []
    for name in names:
        for ref in REFERENCES:
            if name == ref or ref not in set(df["name"]):
                continue
            for obj in OBJECTIVES:
                d, lo, hi = paired_bootstrap(df, name, ref, obj)
                rows.append(
                    {
                        "name": name,
                        "reference": ref,
                        "objective": obj,
                        "difference": d,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "excludes_zero": bool(lo > 0 or hi < 0),
                    }
                )
    return pd.DataFrame(rows)


def fig_disagreement(table, out):
    """The two objectives against each other over the whole design."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    s = ax.scatter(
        table["focal_contribution"],
        table["focal_pool"],
        c=table["mean_p"],
        cmap="viridis",
        s=14,
        alpha=0.85,
    )
    fig.colorbar(s, ax=ax, label="realised mean punishment per member-round")
    for name in REFERENCES:
        r = table[table["name"] == name]
        if len(r):
            ax.scatter(
                r["focal_contribution"], r["focal_pool"], c="red", s=70, marker="*"
            )
            ax.annotate(
                name,
                (r["focal_contribution"].iloc[0], r["focal_pool"].iloc[0]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("focal seat total contribution per round")
    ax.set_ylabel("focal seat common pool per round")
    ax.set_title("The two objectives over the design")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_spend(table, out):
    """Objective against realised spend -- the matched-spend comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, obj in zip(axes, OBJECTIVES):
        ax.scatter(table["mean_p"], table[obj], s=12, alpha=0.6, color="0.5")
        for name, colour in zip(REFERENCES, ("C3", "C0", "C2")):
            r = table[table["name"] == name]
            if len(r):
                ax.scatter(r["mean_p"], r[obj], color=colour, s=80, marker="*")
                ax.annotate(
                    name,
                    (r["mean_p"].iloc[0], r[obj].iloc[0]),
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        ax.set_xlabel("realised mean punishment per member-round")
        ax.set_ylabel(obj.replace("focal_", "focal seat "))
    fig.suptitle("What each objective buys per point of spend")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_profiles(fit_dir, out):
    """One-dimensional cuts through each optimum, both objectives."""
    fig, axes = plt.subplots(2, len(AXES), figsize=(4 * len(AXES), 7))
    for row, obj in enumerate(OBJECTIVES):
        tag = obj.replace("focal_", "")
        prof = pd.read_csv(os.path.join(fit_dir, f"profiles_{tag}.csv"))
        for col, axis in enumerate(AXES):
            ax = axes[row, col]
            p = prof[prof["axis"] == axis]
            ax.plot(p["value"], p["mean"], color=f"C{row}")
            ax.fill_between(
                p["value"],
                p["mean"] - 2 * p["sd"],
                p["mean"] + 2 * p["sd"],
                alpha=0.25,
                color=f"C{row}",
            )
            ax.set_xlabel(axis)
            if col == 0:
                ax.set_ylabel(obj.replace("focal_", "focal seat "))
    fig.suptitle("Posterior mean along each parameter, other parameters at the optimum")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_landscape(fit_dir, out, n=60):
    """The `p_max` x `c0` face of the surrogate at each optimum."""
    optima = pd.read_csv(os.path.join(fit_dir, "optima.csv"))
    fig, axes = plt.subplots(1, len(OBJECTIVES), figsize=(12, 5))
    for ax, (_, o) in zip(np.atleast_1d(axes), optima.iterrows()):
        tag = o["objective"].replace("focal_", "")
        gp = joblib.load(os.path.join(fit_dir, f"gp_{tag}.joblib"))
        u0 = (
            np.array(
                [o["p_max"], o["c0"], np.log10(o["tau"]), o["gamma_ep"], o["gamma_sw"]]
            )
            - LO
        ) / (HI - LO)
        gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        u = np.tile(u0, (n * n, 1))
        u[:, 0], u[:, 1] = gx.ravel(), gy.ravel()
        z = gp.predict(u).reshape(n, n)
        px = LO[0] + gx * (HI[0] - LO[0])
        cy = LO[1] + gy * (HI[1] - LO[1])
        im = ax.contourf(px, cy, z, levels=20, cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.scatter([o["p_max"]], [o["c0"]], color="red", marker="*", s=140)
        ax.set_xlabel("P_max")
        ax.set_ylabel("c0")
        ax.set_title(f"{tag}: surrogate at the fitted optimum")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_policy_shape(shape, out):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name in shape.columns:
        ax.plot(list(RPA_LABELS), shape[name], marker="o", label=name)
    ax.set_xlabel("contribution bin (evaluation suite bins)")
    ax.set_ylabel("mean punishment")
    ax.set_title("Policy shape: what each manager punishes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run(args):
    os.makedirs(args.out, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fit_seeds = [int(s) for s in args.fit_seeds.split(",")]
    holdout = [int(s) for s in args.holdout_seeds.split(",")]

    design = pd.read_csv(args.design)
    design_ep = load_sweep(args.design_run)
    design_tab = design.merge(
        per_point(design_ep, seeds=fit_seeds), on="name", how="inner"
    )
    design_tab.to_csv(os.path.join(args.out, "design_summary.csv"), index=False)

    valid_design = pd.read_csv(args.valid_design)
    valid_ep = load_sweep(args.valid_run)
    vt = validation_table(valid_ep, valid_design, holdout)
    vt.to_csv(os.path.join(args.out, "validation_table.csv"), index=False)
    contrasts(valid_ep, valid_design, holdout).to_csv(
        os.path.join(args.out, "contrasts.csv"), index=False
    )

    shape_src = per_point(valid_ep[valid_ep["seed"].isin(holdout)])
    shape = {}
    for name in SHAPE_ROWS:
        r = shape_src[shape_src["name"] == name]
        if len(r):
            shape[name] = [float(r[f"rpa_{lab}"].iloc[0]) for lab in RPA_LABELS]
    shape["human"] = human_policy_shape(args.human).to_numpy()
    shape = pd.DataFrame(shape, index=list(RPA_LABELS))
    shape.to_csv(os.path.join(args.out, "policy_shape.csv"))

    fig_disagreement(design_tab, os.path.join(args.out, "objective_disagreement.jpg"))
    fig_spend(design_tab, os.path.join(args.out, "objective_vs_spend.jpg"))
    fig_profiles(args.fit_dir, os.path.join(args.out, "surrogate_profiles.jpg"))
    fig_landscape(args.fit_dir, os.path.join(args.out, "surrogate_landscape.jpg"))
    fig_policy_shape(shape, os.path.join(args.out, "policy_shape.jpg"))
    print(vt.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design-run", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--valid-run", required=True)
    ap.add_argument("--valid-design", required=True)
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--fit-seeds", default="42,43")
    ap.add_argument("--holdout-seeds", default="44,45,46")
    ap.add_argument("--human", default="experiments/2group_8agent_50ep.csv")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
