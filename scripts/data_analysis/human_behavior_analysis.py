"""Interpretable models of human behaviour in the 2g8a group-switching game.

Goal: understand whether human play is *sensible*. Fits simple, explainable
models (standardised OLS / logistic regression) + event studies for:

  Q1 what drives contributions
  Q2 what drives switching
  Q3 what drives punishment
  Q4 how people adapt to a new group after switching
  Q5 evolution over rounds / end-game effects
  + bonus: does punishment deter?  is switching smart?

Data is the pair-augmented CSV; we DE-DUPLICATE to the 50 real episodes (each
pair is stored twice with group labels swapped — identical contributions).

Usage:
    .venv/bin/python scripts/data_analysis/human_behavior_analysis.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/2group_8agent_50ep.csv"
OUT = ROOT / "plots/data_analysis/human_behavior"
OUT.mkdir(parents=True, exist_ok=True)
DECISIONS = {4, 8, 12, 16, 20}


def load():
    df = pd.read_csv(DATA)
    df = df[df["experiment_name"] == "ah_group_switching"].copy()
    # de-duplicate the pair augmentation: keep one episode per pair_id
    keep = df.groupby("pair_id")["episode_id"].transform("min")
    df = df[df["episode_id"] == keep].copy()
    df = df.sort_values(["episode_id", "player_id", "round_number"]).reset_index(drop=True)
    # per-player lags
    g = df.groupby(["episode_id", "player_id"])
    df["self_prev_c"] = g["contribution"].shift(1)
    df["self_prev_p"] = g["punishment"].shift(1)
    df["prev_group"] = g["group_id"].shift(1)
    df["switch"] = df["prev_group"].notna() & df["prev_group"].ne(df["group_id"])
    # group aggregates per (episode, round, group)
    gr = df.groupby(["episode_id", "round_number", "group_id"])["contribution"]
    gsum, gcnt = gr.transform("sum"), gr.transform("count")
    df["own_grp_mean_c"] = (gsum - df["contribution"]) / (gcnt - 1)  # same round, LOO
    gmean = df.groupby(["episode_id", "round_number", "group_id"])["contribution"].mean()
    cg = df.groupby(["episode_id", "round_number", "group_id"])["common_good"].first()
    gmean_d, cg_d = gmean.to_dict(), cg.to_dict()
    gsum_d = df.groupby(["episode_id", "round_number", "group_id"])["contribution"].sum().to_dict()
    gcnt_d = df.groupby(["episode_id", "round_number", "group_id"])["contribution"].count().to_dict()

    def look(d, ep, r, gid, default=np.nan):
        return d.get((ep, r, gid), default)

    # signals observed last round (group the player was in at t-1)
    own_prev, oth_prev, own_cg_p, oth_cg_p = [], [], [], []
    for ep, r, pg, sc in df[["episode_id", "round_number", "prev_group", "self_prev_c"]].values:
        if pd.isna(pg):
            own_prev.append(np.nan); oth_prev.append(np.nan)
            own_cg_p.append(np.nan); oth_cg_p.append(np.nan); continue
        pg = int(pg); pr = r - 1
        s, c = look(gsum_d, ep, pr, pg), look(gcnt_d, ep, pr, pg)
        own_prev.append((s - sc) / (c - 1) if c and c > 1 else np.nan)
        oth_prev.append(look(gmean_d, ep, pr, 1 - pg))
        own_cg_p.append(look(cg_d, ep, pr, pg))
        oth_cg_p.append(look(cg_d, ep, pr, 1 - pg))
    df["own_grp_prev_mean_c"] = own_prev
    df["oth_grp_prev_mean_c"] = oth_prev
    df["own_cg_prev"] = own_cg_p
    df["oth_cg_prev"] = oth_cg_p
    return df


def stdz(X):
    return (X - X.mean()) / X.std()


def ols(df, feats, tgt):
    A = df.dropna(subset=feats + [tgt])
    X = stdz(A[feats]).copy(); X.insert(0, "const", 1.0)
    y = stdz(A[tgt]).values
    b, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    pred = X.values @ b
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r2, dict(zip(feats, b[1:])), len(A)


def logit(df, feats, tgt):
    A = df.dropna(subset=feats + [tgt])
    X = stdz(A[feats]).values
    y = A[tgt].astype(int).values
    m = LogisticRegression(max_iter=1000).fit(X, y)
    # McFadden pseudo-R2
    p = m.predict_proba(X)[:, 1].clip(1e-6, 1 - 1e-6)
    ll = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    pbar = y.mean()
    ll0 = (y * np.log(pbar) + (1 - y) * np.log(1 - pbar)).sum()
    return 1 - ll / ll0, dict(zip(feats, m.coef_[0])), len(A)


def coef_panel(ax, betas, title, xlabel):
    ks = list(betas)[::-1]
    vs = [betas[k] for k in ks]
    cols = ["#2ca02c" if v >= 0 else "#d62728" for v in vs]
    ax.barh(ks, vs, color=cols)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    for i, v in enumerate(vs):
        ax.text(v, i, f" {v:+.2f}", va="center", fontsize=8)


def main():
    df = load()
    print(f"De-duplicated: {df['episode_id'].nunique()} episodes, {len(df)} rows\n")

    # ---- Q1 contribution ----
    f1 = ["self_prev_c", "self_prev_p", "own_grp_prev_mean_c", "oth_grp_prev_mean_c",
          "round_number", "switch"]
    df["switch_f"] = df["switch"].astype(float)
    f1m = [x if x != "switch" else "switch_f" for x in f1]
    r2_1, b1, n1 = ols(df[df["round_number"] >= 1], f1m, "contribution")
    print(f"Q1 CONTRIBUTION  (R2={r2_1:.3f}, N={n1})")
    for k in f1m:
        print(f"   {k:22s} {b1[k]:+.3f}")

    # ---- Q2 switching (decision rounds only) ----
    dec = df[df["round_number"].isin(DECISIONS) & (df["selection_timeout"].fillna(0) == 0)].copy()
    dec["gap_cg"] = dec["oth_cg_prev"] - dec["own_cg_prev"]
    f2 = ["own_cg_prev", "oth_cg_prev", "self_prev_p", "self_prev_c", "round_number"]
    r2_2, b2, n2 = logit(dec, f2, "switch")
    print(f"\nQ2 SWITCHING  (pseudo-R2={r2_2:.3f}, N={n2}, switch rate={dec['switch'].mean():.1%})")
    for k in f2:
        print(f"   {k:22s} {b2[k]:+.3f}  (log-odds per SD)")
    rg, bg, _ = logit(dec, ["gap_cg", "self_prev_p", "round_number"], "switch")
    print(f"   [gap model] gap_cg {bg['gap_cg']:+.3f}  pseudo-R2={rg:.3f}")

    # ---- Q3 punishment (manager reacts to same-round contribution) ----
    pun = df[(df["manager_no_input"] == 0) & (df["player_no_input"] == 0)].copy()
    f3 = ["contribution", "own_grp_mean_c", "self_prev_p", "round_number"]
    r2_3, b3, n3 = ols(pun, f3, "punishment")
    print(f"\nQ3 PUNISHMENT  (R2={r2_3:.3f}, N={n3})")
    for k in f3:
        print(f"   {k:22s} {b3[k]:+.3f}")

    # ---- coef figure ----
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    coef_panel(ax[0], {k: b1[k] for k in f1m}, "Q1  contribution", "std β")
    coef_panel(ax[1], b2, "Q2  switch (log-odds/SD)", "logit coef")
    coef_panel(ax[2], b3, "Q3  punishment", "std β")
    fig.suptitle("What drives contribution, switching, punishment", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "drivers.png", dpi=130); print(f"\nsaved {OUT/'drivers.png'}")

    # ---- Q4 adaptation after switch ----
    # for each switch event at decision round t:
    #   pre        = own contribution at t-1 (in old group)
    #   new_peers  = new group's leave-one-out mean at t (own_grp_mean_c)
    #   post       = own contribution at t (in new group)
    # group levels observed at the decision (t-1) for the smart-switch test.
    gmean = df.groupby(["episode_id", "round_number", "group_id"])["contribution"].mean().to_dict()
    sw_rows = df[df["switch"]][
        ["episode_id", "round_number", "prev_group", "group_id",
         "self_prev_c", "contribution", "own_grp_mean_c"]]
    rows = []
    for ep, r, pg, gid, sc, c, peers in sw_rows.values:
        rows.append(dict(
            pre=sc, post=c, new_peers=peers,
            new_pre=gmean.get((ep, r - 1, int(gid))),   # joined group level, pre-decision
            old_pre=gmean.get((ep, r - 1, int(pg))),    # left group level, pre-decision
        ))
    AD = pd.DataFrame(rows)
    ADr = AD.dropna(subset=["pre", "post", "new_peers"])
    r2_4, b4, n4 = ols(ADr, ["pre", "new_peers"], "post")
    print(f"\nQ4 ADAPTATION after switch  (N={n4})")
    print(f"   post-switch contribution ~ own pre + new-group peer mean (std β):")
    print(f"     own pre-switch    {b4['pre']:+.3f}")
    print(f"     new-group peers   {b4['new_peers']:+.3f}   (>0 => adopts new group's norm)")
    print(f"   raw: switchers post mean={ADr['post'].mean():.2f}, "
          f"new peers={ADr['new_peers'].mean():.2f}, own pre={ADr['pre'].mean():.2f}")

    # ---- Q5 round evolution ----
    by = df.groupby("round_number")
    mc = by["contribution"].mean()
    mp = pun.groupby("round_number")["punishment"].mean()
    sw = df[df["round_number"].isin(DECISIONS)].groupby("round_number")["switch"].mean()
    cgb = df.groupby(["episode_id", "round_number", "group_id"])["common_good"].first()\
            .groupby("round_number").mean()
    print("\nQ5 EVOLUTION / END-GAME")
    print(f"   mean contribution round 0={mc.iloc[0]:.2f}, mid={mc.loc[12]:.2f}, "
          f"last(23)={mc.iloc[-1]:.2f}, drop 22->23={mc.iloc[-1]-mc.loc[22]:+.2f}")
    print(f"   switch rate first(4)={sw.loc[4]:.1%} -> last(20)={sw.loc[20]:.1%}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    ax[0].plot(mc.index, mc.values, "-o", ms=3); ax[0].set_title("Mean contribution")
    ax[0].set_xlabel("round"); ax[0].axvline(23, color="grey", ls=":", lw=0.8)
    for d in DECISIONS:
        ax[0].axvline(d, color="orange", ls=":", lw=0.5)
    ax[1].plot(mp.index, mp.values, "-o", ms=3, color="#d62728"); ax[1].set_title("Mean punishment")
    ax[1].set_xlabel("round")
    ax[2].plot(sw.index, sw.values, "-o", ms=4, color="#9467bd"); ax[2].set_title("Switch rate (decisions)")
    ax[2].set_xlabel("round"); ax[2].set_ylim(0, None)
    fig.suptitle("Evolution over rounds (orange = switch decisions, dotted grey = final round)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "evolution.png", dpi=130); print(f"saved {OUT/'evolution.png'}")

    # ---- bonus: does punishment deter? Δc(t->t+1) ~ punishment(t) | c(t) ----
    df["next_c"] = df.groupby(["episode_id", "player_id"])["contribution"].shift(-1)
    df["dc"] = df["next_c"] - df["contribution"]
    det = df.dropna(subset=["dc", "punishment", "contribution"])
    r2_d, bd, nd = ols(det, ["punishment", "contribution"], "next_c")
    print(f"\nBONUS deterrence: next_c ~ punishment + c  (N={nd})")
    print(f"   punishment(t) {bd['punishment']:+.3f}  (>0 => punishment raises next contribution)")

    # ---- bonus: is switching smart? pre-decision quality of joined vs left ----
    smart = AD.dropna(subset=["new_pre", "old_pre"])
    better = (smart["new_pre"] > smart["old_pre"]).mean()
    print(f"\nBONUS smart switching: joined group was contributing more than the left "
          f"group (as of the decision) in {better:.1%} of switches  "
          f"(joined {smart['new_pre'].mean():.2f} vs left {smart['old_pre'].mean():.2f})")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    # deterrence: binned
    det2 = det.copy()
    det2["pbin"] = pd.cut(det2["punishment"], [-0.1, 0.1, 5, 15, 31],
                          labels=["0", "1-5", "6-15", "16-30"])
    g = det2.groupby("pbin", observed=True)["dc"].mean()
    ax[0].bar(g.index.astype(str), g.values, color="#1f77b4")
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_title("Punishment -> next-round Δcontribution")
    ax[0].set_xlabel("punishment received"); ax[0].set_ylabel("Δ contribution")
    ax[1].bar(["left group", "joined group"], [smart["old_pre"].mean(), smart["new_pre"].mean()],
              color=["#d62728", "#2ca02c"])
    ax[1].set_title("Switchers: quality of group left vs joined"); ax[1].set_ylabel("mean contribution")
    fig.tight_layout()
    fig.savefig(OUT / "bonus.png", dpi=130); print(f"saved {OUT/'bonus.png'}")


if __name__ == "__main__":
    main()
