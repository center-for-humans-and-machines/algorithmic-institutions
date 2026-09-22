"""Expressiveness probe for the switch and punishment AHs.

Companion to `expressiveness_group_disentanglement.py` (contribution AH) and the
report `reports/expressiveness_switch_and_punishment_50ep.md`. Same perspective —
group-average influence and own-vs-other disentanglement — applied to:

  - switch_predictor/opt_50ep_doubled.yml      (target does_switch)
  - punishment/rnn_edge_50ep_doubled.yml       (target punishment)

Prints the numbers and saves one summary figure (own- vs other-group partial
effect across the three AH targets) to plots/data_analysis/.

Usage:
    .venv/bin/python scripts/data_analysis/expressiveness_switch_punishment.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/2group_8agent_50ep.csv"
ART = ROOT / "artifacts/artificial_humans"
OUTDIR = ROOT / "plots/data_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)


def stdz(X):
    return (X - X.mean()) / X.std()


def ols(A, feats, tgt):
    X = stdz(A[feats]).copy()
    X.insert(0, "const", 1.0)
    y = stdz(A[tgt]).values
    b, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    pred = X.values @ b
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r2, dict(zip(X.columns, b))


def switch_feature_importance():
    p = (ART / "switch_pred_opt_50ep_doubled/metrics/"
         "architecture_mlp+rnn+edge__dataset_50ep_doubled.parquet")
    m = pd.read_parquet(p)
    ll = m[(m["set"] == "test") & (m["name"] == "log_loss")]
    llf = ll[ll["epoch"] == ll["epoch"].max()]
    base = llf[llf["shuffle_feature"].isna()
               & llf["leave_one_in_shuffle_feature"].isna()]
    bm = base.groupby("cv_split")["value"].mean().mean()
    print(f"\n# SWITCH feature importance (baseline test log_loss = {bm:.4f})")
    for f, g in llf[llf["shuffle_feature"].notna()].groupby("shuffle_feature"):
        v = g.groupby("cv_split")["value"].mean().mean()
        print(f"  shuffle {f:18s} Δ={v - bm:+.4f}")


def load():
    df = pd.read_csv(DATA)
    df = df[df["experiment_name"] == "ah_group_switching"].copy()
    return df.sort_values(["episode_id", "player_id", "round_number"])


def switch_structure(df):
    cg = df.groupby(["episode_id", "round_number", "group_id"])["common_good"].first()
    cgmap = cg.to_dict()
    df["prev_group"] = df.groupby(["episode_id", "player_id"])["group_id"].shift(1)
    rows = []
    for _, r in df.iterrows():
        t = r["round_number"]
        if t == 0 or t % 4 != 0 or r.get("selection_timeout", 0) == 1:
            continue
        pg = r["prev_group"]
        if pd.isna(pg):
            continue
        pg = int(pg)
        own = cgmap.get((r["episode_id"], t - 1, pg))
        oth = cgmap.get((r["episode_id"], t - 1, 1 - pg))
        if own is None or oth is None:
            continue
        rows.append(dict(sw=int(r["group_id"] != pg), own_cg=own, oth_cg=oth))
    S = pd.DataFrame(rows)
    r2, b = ols(S, ["own_cg", "oth_cg"], "sw")
    print(f"\n# SWITCH structure (N={len(S)}, switch rate={S['sw'].mean():.1%})")
    print(f"  corr(switch, gap=oth-own)        = "
          f"{S['sw'].corr(S['oth_cg'] - S['own_cg']):+.3f}")
    print(f"  std OLS switch ~ own_cg + oth_cg  R2={r2:.3f}  "
          f"own={b['own_cg']:+.3f}  oth={b['oth_cg']:+.3f}")
    print(f"  separability corr(own_cg, oth_cg) = {S['own_cg'].corr(S['oth_cg']):+.3f}")
    return b["own_cg"], b["oth_cg"]


def punishment_structure(df):
    gc = df.groupby(["episode_id", "round_number", "group_id"])["contribution"]
    df["own_loo_mean_c"] = (gc.transform("sum") - df["contribution"]) / (
        gc.transform("count") - 1
    )
    om = df.groupby(["episode_id", "round_number", "group_id"])["contribution"].mean()
    ommap = om.to_dict()
    df["oth_mean_c"] = [
        ommap.get((e, r, 1 - g), np.nan)
        for e, r, g in df[["episode_id", "round_number", "group_id"]].values
    ]
    df["prev_contribution"] = df.groupby(["episode_id", "player_id"])[
        "contribution"
    ].shift(1)
    P = df[(df["manager_no_input"] == 0) & (df["player_no_input"] == 0)].dropna(
        subset=["punishment", "contribution", "own_loo_mean_c", "oth_mean_c"]
    )
    print(f"\n# PUNISHMENT structure (N={len(P)})")
    print(f"  corr(punish, same-round contribution) = "
          f"{P['punishment'].corr(P['contribution']):+.3f}")
    print(f"  corr(punish, prev-round contribution) = "
          f"{P['punishment'].corr(P['prev_contribution']):+.3f}  "
          f"(model uses prev_contribution)")
    r2, b = ols(P, ["contribution", "own_loo_mean_c", "oth_mean_c"], "punishment")
    print(f"  std OLS punish ~ self_c + own_mean + oth_mean  R2={r2:.3f}")
    for k, lbl in [("contribution", "self_c"), ("own_loo_mean_c", "own_mean"),
                   ("oth_mean_c", "oth_mean")]:
        print(f"      {lbl:9s} {b[k]:+.3f}")
    return b["own_loo_mean_c"], b["oth_mean_c"]


def summary_figure(sw, pun):
    # contribution numbers come from the contribution report (§5a)
    models = ["contribution\n(c ~ own/other avg c)",
              "switch\n(switch ~ own/other cg)",
              "punishment\n(punish ~ own/other mean c)"]
    own = [0.19, sw[0], pun[0]]
    oth = [0.00, sw[1], pun[1]]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - w / 2, own, w, label="own-group", color="#2ca02c")
    ax.bar(x + w / 2, oth, w, label="other-group", color="#d62728")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("standardised partial coefficient")
    ax.set_title("The other group matters only for switching")
    ax.legend()
    for xi, (o, t) in enumerate(zip(own, oth)):
        ax.text(xi - w / 2, o, f"{o:+.2f}", ha="center",
                va="bottom" if o >= 0 else "top", fontsize=8)
        ax.text(xi + w / 2, t, f"{t:+.2f}", ha="center",
                va="bottom" if t >= 0 else "top", fontsize=8)
    fig.tight_layout()
    f = OUTDIR / "expressiveness_own_vs_other_by_target.png"
    fig.savefig(f, dpi=130)
    print(f"\nsaved {f}")


if __name__ == "__main__":
    switch_feature_importance()
    df = load()
    sw = switch_structure(df)
    pun = punishment_structure(load())
    summary_figure(sw, pun)
