"""Human pilots: dose-response of contribution to punishment, at lags 1-3, and by round."""
import pandas as pd, numpy as np
d = pd.read_csv("experiments/pilot_random1_player_round_slim.csv")
d = d.sort_values(["episode_id", "participant_code", "round_number"])
d.loc[d.player_no_input == 1, "contribution"] = np.nan
g = d.groupby(["episode_id", "participant_code"])
for k in (1, 2, 3):
    d[f"dc{k}"] = g.contribution.shift(-k) - d.contribution
d["others"] = (d.groupby(["episode_id", "round_number"]).contribution.transform("sum") - d.contribution) / 3
d["cbin"] = pd.cut(d.contribution, [-1, 5, 10, 15, 20])
d["pbin"] = pd.cut(d.punishment, [-1, 0, 3, 7, 30])
for k in (1, 2, 3):
    print(f"lag {k} mean dc")
    print(d.pivot_table(index="cbin", columns="pbin", values=f"dc{k}", aggfunc="mean", observed=False).round(2))
print(d.pivot_table(index="cbin", columns="pbin", values="dc1", aggfunc="size", observed=False))
print("by round and experiment: mean c, mean p")
print(d.pivot_table(index="round_number", columns="experiment_name", values=["contribution", "punishment"], aggfunc="mean").round(2))
# regression dc1 ~ p + c + others  for c<=15
import numpy.linalg as la
s = d.dropna(subset=["dc1", "others"])
X = np.c_[np.ones(len(s)), s.punishment, s.contribution, s.others, s.punishment * (s.contribution > 15)]
b = la.lstsq(X, s.dc1, rcond=None)[0]
print("dc1 ~ 1 + p + c + others + p*(c>15):", b.round(3))
s = d.dropna(subset=["dc3", "others"])
X = np.c_[np.ones(len(s)), s.punishment, s.contribution, s.others, s.punishment * (s.contribution > 15)]
print("dc3 ~ same:", la.lstsq(X, s.dc3, rcond=None)[0].round(3))
print("timeout rate", d.player_no_input.mean())
