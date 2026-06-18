"""Expressiveness probe: own-group vs other-group contribution disentanglement.

Supports the report `reports/expressiveness_group_switching_contribution_50ep.md`.

Two empirical questions about the contribution AH trained by
`configs/training/artificial_humans/contribution/group_switching_contribution_50ep.yml`:

1. Does the TRAINED model actually use group identity? (feature importance
   from the training metrics parquet -- shuffle / leave-one-in of `agent_group`.)
2. Is the own- vs other-group signal even SEPARABLE and PRESENT in the data?
   (group-size variability, own/other correlation, and a standardised OLS of
   focal contribution on self + own-group avg + other-group avg.)

Outputs printed numbers and two figures under plots/data_analysis/.

Usage:
    .venv/bin/python scripts/data_analysis/expressiveness_group_disentanglement.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    ROOT
    / "artifacts/artificial_humans/group_switching_contribution_50ep/metrics/"
    "architecture_node+edge+rnn__dataset_50ep__epochs_575.parquet"
)
DATA = ROOT / "experiments/2group_8agent_50ep.csv"
OUTDIR = ROOT / "plots/data_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)


def feature_importance():
    """CV-averaged, final-epoch test log_loss: baseline vs perturbations."""
    m = pd.read_parquet(METRICS)
    ll = m[(m["set"] == "test") & (m["name"] == "log_loss")].copy()
    final = int(ll["epoch"].max())
    llf = ll[ll["epoch"] == final]

    base = llf[
        llf["shuffle_feature"].isna() & llf["leave_one_in_shuffle_feature"].isna()
    ]
    base_mean = base.groupby("cv_split")["value"].mean().mean()

    def cv(g):
        return g.groupby("cv_split")["value"].mean().mean()

    shuffle = {
        f: cv(g) - base_mean
        for f, g in llf[llf["shuffle_feature"].notna()].groupby("shuffle_feature")
    }
    keeponly = {
        f: cv(g) - base_mean
        for f, g in llf[llf["leave_one_in_shuffle_feature"].notna()].groupby(
            "leave_one_in_shuffle_feature"
        )
    }
    print(f"\n# Feature importance  (baseline test log_loss = {base_mean:.4f})")
    print("shuffle ONE feature (rise in log_loss = importance):")
    for f, d in sorted(shuffle.items(), key=lambda x: -x[1]):
        print(f"  {f:18s} {d:+.4f}")
    print("keep ONLY this feature, shuffle the rest:")
    for f, d in sorted(keeponly.items(), key=lambda x: -x[1]):
        print(f"  {f:18s} {d:+.4f}")
    return base_mean, shuffle, keeponly


def behavioural_structure():
    df = pd.read_csv(DATA)
    df = df[df["experiment_name"] == "ah_group_switching"].copy()
    df = df.sort_values(["episode_id", "player_id", "round_number"])
    df["prev_c"] = df.groupby(["episode_id", "player_id"])["contribution"].shift(1)

    gs = df.groupby(["episode_id", "round_number", "group_id"])["player_id"].nunique()
    print("\n# Group-size distribution (players per group per round)")
    print(gs.value_counts().sort_index().to_string())
    print(
        f"mean {gs.mean():.2f}  std {gs.std():.2f}  min {gs.min()}  max {gs.max()}"
    )

    rows = []
    for (_, _), sub in df[df["round_number"] >= 1].groupby(
        ["episode_id", "round_number"]
    ):
        sub = sub.dropna(subset=["prev_c"])
        if sub["group_id"].nunique() < 2:
            continue
        for _, r in sub.iterrows():
            g, pid = r["group_id"], r["player_id"]
            own = sub[(sub["group_id"] == g) & (sub["player_id"] != pid)]["prev_c"]
            oth = sub[sub["group_id"] != g]["prev_c"]
            if len(own) == 0 or len(oth) == 0:
                continue
            rows.append(
                dict(
                    c=r["contribution"],
                    prev_c=r["prev_c"],
                    own_avg=own.mean(),
                    oth_avg=oth.mean(),
                    own_n=len(own) + 1,
                )
            )
    A = pd.DataFrame(rows)
    print(f"\n# Behavioural structure  (N focal-rounds = {len(A)})")
    print(f"corr(own_avg, oth_avg)      = {A['own_avg'].corr(A['oth_avg']):.3f}")
    print(f"corr(c, prev_c)             = {A['c'].corr(A['prev_c']):.3f}")
    print(f"corr(c, own_avg)            = {A['c'].corr(A['own_avg']):.3f}")
    print(f"corr(c, oth_avg)            = {A['c'].corr(A['oth_avg']):.3f}")
    print(f"own group size != 4 share   = {(A['own_n'] != 4).mean():.1%}")

    X = A[["prev_c", "own_avg", "oth_avg"]]
    Xs = (X - X.mean()) / X.std()
    Xs.insert(0, "const", 1.0)
    y = (A["c"] - A["c"].mean()) / A["c"].std()
    beta, *_ = np.linalg.lstsq(Xs.values, y.values, rcond=None)
    betas = dict(zip(Xs.columns, beta))
    print("standardised OLS  c ~ prev_c + own_avg + oth_avg:")
    for k in ["prev_c", "own_avg", "oth_avg"]:
        print(f"  {k:10s} {betas[k]:+.3f}")
    return A, betas


def make_figures(shuffle, betas, A):
    # Fig 1: feature importance
    fig, ax = plt.subplots(figsize=(6, 3.2))
    order = ["prev_contribution", "prev_punishment", "agent_group"]
    vals = [shuffle.get(k, np.nan) for k in order]
    colors = ["#d62728" if k == "agent_group" else "#1f77b4" for k in order]
    ax.barh(order, vals, color=colors)
    ax.set_xlabel("Δ test log-loss when feature is shuffled (importance)")
    ax.set_title("Trained model: group identity is essentially unused")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.4f}", va="center", fontsize=9)
    ax.axvline(0, color="k", lw=0.6)
    fig.tight_layout()
    f1 = OUTDIR / "expressiveness_feature_importance.png"
    fig.savefig(f1, dpi=130)
    print(f"\nsaved {f1}")

    # Fig 2: behavioural partial effects
    fig, ax = plt.subplots(figsize=(6, 3.2))
    order = ["prev_c", "own_avg", "oth_avg"]
    labels = ["focal prev contribution\n(self)", "own-group avg", "other-group avg"]
    vals = [betas[k] for k in order]
    colors = ["#7f7f7f", "#2ca02c", "#d62728"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("standardised partial coefficient")
    ax.set_title("What actually drives contribution (data)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:+.2f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    f2 = OUTDIR / "expressiveness_behavioural_effects.png"
    fig.savefig(f2, dpi=130)
    print(f"saved {f2}")


def candidate_features():
    """§7: incremental value of own-group avg contribution / common good /
    avg punishment for predicting the focal's next contribution."""
    df = pd.read_csv(DATA)
    df = df[df["experiment_name"] == "ah_group_switching"].copy()
    df = df.sort_values(["episode_id", "player_id", "round_number"])
    for col in ["contribution", "punishment", "common_good"]:
        df[f"prev_{col}"] = df.groupby(["episode_id", "player_id"])[col].shift(1)
    pg = df.groupby(["episode_id", "player_id"])["group_id"].shift(1)
    df["does_switch"] = (pg.notna() & pg.ne(df["group_id"])).astype(float)

    rows = []
    for (_, _), sub in df[df["round_number"] >= 1].groupby(
        ["episode_id", "round_number"]
    ):
        sub = sub.dropna(subset=["prev_contribution"])
        if sub["group_id"].nunique() < 2:
            continue
        for _, r in sub.iterrows():
            gid, pid = r["group_id"], r["player_id"]
            own = sub[(sub["group_id"] == gid) & (sub["player_id"] != pid)]
            if len(own) == 0:
                continue
            rows.append(
                dict(
                    c=r["contribution"],
                    self_c=r["prev_contribution"],
                    own_avg_c=own["prev_contribution"].mean(),
                    own_cg=r["prev_common_good"],
                    own_avg_p=own["prev_punishment"].mean(),
                    sw=r["does_switch"],
                )
            )
    A = pd.DataFrame(rows).dropna()

    def stdz(X):
        return (X - X.mean()) / X.std()

    y = stdz(A["c"]).values

    def r2(feats):
        X = stdz(A[feats]).copy()
        X.insert(0, "const", 1.0)
        beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
        pred = X.values @ beta
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    cands = [("avg contribution", "own_avg_c"),
             ("common good", "own_cg"),
             ("avg punishment", "own_avg_p")]
    print(f"\n# Candidate own-group features  (N = {len(A)})")
    print("(1) ALONE (univariate R^2):")
    print(f"  {'self':20s} {r2(['self_c']):.4f}")
    for name, col in cands:
        print(f"  {name:20s} {r2([col]):.4f}")
    base = r2(["self_c"])
    print(f"(2) +SELF, each individually (self={base:.4f}):")
    for name, col in cands:
        print(f"  {name:20s} ΔR2={r2(['self_c', col]) - base:+.4f}")
    print("(3) CUMULATIVE (self -> +avg_c -> +cg -> +avg_p):")
    prev, feats = 0.0, ["self_c"]
    for name, col in [("self_c", "self_c")] + cands:
        feats = feats if col == "self_c" else feats + [col]
        v = r2(feats)
        print(f"  {name:20s} R2={v:.4f}  ΔR2={v - prev:+.4f}")
        prev = v
    full = ["self_c", "own_avg_c", "own_cg", "own_avg_p"]
    rf = r2(full)
    print("(4) UNIQUE (drop from full self+all-three):")
    for name, col in cands:
        print(f"  {name:20s} ΔR2={r2([x for x in full if x != col]) - rf:+.4f}")
    print(f"collinearity corr(own_cg, own_avg_c) = "
          f"{A['own_cg'].corr(A['own_avg_c']):.2f}")
    # §8: does switching move contribution?
    r_sw = r2(["self_c", "own_avg_c", "sw"])
    r_no = r2(["self_c", "own_avg_c"])
    print(
        f"(5) switch effect: share={A['sw'].mean():.1%}  "
        f"switched_c={A.loc[A.sw == 1, 'c'].mean():.2f}  "
        f"stayed_c={A.loc[A.sw == 0, 'c'].mean():.2f}  "
        f"ΔR2(+does_switch)={r_sw - r_no:+.4f}"
    )


if __name__ == "__main__":
    _, shuffle, _ = feature_importance()
    A, betas = behavioural_structure()
    candidate_features()
    make_figures(shuffle, betas, A)
