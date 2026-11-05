"""Humanlike sims by episode: cheap vs costly punishers; early punishment vs later contributions."""
import pandas as pd, numpy as np
df = pd.read_parquet("plots/simulation/29_EXP1_reference/per_round.parquet")
df["mgr"] = df.run.str.split("managed by ").str[-1]
v = df[df.contribution_valid]
r = v.groupby(["mgr", "episode", "round_number"]).agg(c=("contribution", "mean"), p=("punishment", "mean"), pay=("payoff", "mean")).reset_index()
r["half"] = np.where(r.round_number < 8, "early", np.where(r.round_number < 16, "mid", "late"))
e = r.pivot_table(index=["mgr", "episode"], columns="half", values=["c", "p", "pay"], aggfunc="mean")
e.columns = [f"{a}_{b}" for a, b in e.columns]
e["c0"] = r[r.round_number == 0].set_index(["mgr", "episode"]).c
e["score"] = r.groupby(["mgr", "episode"]).pay.sum()
h = e.loc["humanlike"].copy(); z = e.loc["zero"].copy()
h["c0bin"] = pd.cut(h.c0, [-1, 8, 12, 16, 20]); z["c0bin"] = pd.cut(z.c0, [-1, 8, 12, 16, 20])
h["pbin"] = pd.qcut(h.p_early, 4)
print(h.groupby(["c0bin", "pbin"], observed=True)[["p_early", "c_mid", "c_late", "score"]].mean().round(2))
print("zero", z.groupby("c0bin", observed=True)[["c_mid", "c_late", "score"]].agg(["mean", "size"]).round(2))
