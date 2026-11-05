"""Reference sims: how next contribution responds to punishment, given own contribution."""
import pandas as pd, numpy as np
df = pd.read_parquet("plots/simulation/29_EXP1_reference/per_round.parquet")
df["mgr"] = df.run.str.split("managed by ").str[-1]
df = df.sort_values(["mgr", "episode", "participant_code", "round_number"])
df.loc[~df.contribution_valid, "contribution"] = np.nan
g = df.groupby(["mgr", "episode", "participant_code"])
df["c_next"] = g.contribution.shift(-1)
df["dc"] = df.c_next - df.contribution
# group mean of others
gm = df.groupby(["mgr", "episode", "round_number"]).contribution.transform("mean")
df["gmean"] = gm
print("timeout rate", df.groupby("mgr").contribution_valid.apply(lambda v: 1 - v.mean()))
tot = df[df.contribution_valid].groupby(["mgr", "episode", "round_number"]).payoff.mean().groupby("mgr").sum() / 1000
print("score (sum mean payoff / episodes)", tot)
h = df[(df.mgr == "humanlike") & df.contribution_valid & df.c_next.notna()].copy()
h["cbin"] = pd.cut(h.contribution, [-1, 0, 5, 10, 15, 19, 20])
h["pbin"] = pd.cut(h.punishment, [-1, 0, 2, 5, 10, 30])
print(h.pivot_table(index="cbin", columns="pbin", values="dc", aggfunc="mean").round(2))
print(h.pivot_table(index="cbin", columns="pbin", values="dc", aggfunc="size"))
z = df[(df.mgr == "zero") & df.contribution_valid & df.c_next.notna()].copy()
z["cbin"] = pd.cut(z.contribution, [-1, 0, 5, 10, 15, 19, 20])
print("zero dc by cbin", z.groupby("cbin").dc.agg(["mean", "size"]).round(2))
# humanlike punishment policy vs contribution
h["dev"] = h.contribution - h.gmean
print(h.groupby("cbin").punishment.mean().round(2))
