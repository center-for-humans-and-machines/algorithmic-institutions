"""Humanlike sims: persistence of the punishment effect (lags 1..8) and spillover to others."""
import pandas as pd, numpy as np, numpy.linalg as la
df = pd.read_parquet("plots/simulation/29_EXP1_reference/per_round.parquet")
df = df[df.run.str.endswith("humanlike")].copy()
df.loc[~df.contribution_valid, "contribution"] = np.nan
df = df.sort_values(["episode", "participant_code", "round_number"])
g = df.groupby(["episode", "participant_code"])
grp = df.groupby(["episode", "round_number"])
df["others"] = (grp.contribution.transform("sum") - df.contribution) / (grp.contribution.transform("count") - 1)
df["P_others"] = (grp.punishment.transform("sum") - df.punishment) / 3
for k in range(1, 9):
    df["dc"] = g.contribution.shift(-k) - df.contribution
    s = df.dropna(subset=["dc", "others"])
    lo = s.contribution <= 10
    X = np.c_[np.ones(len(s)), s.punishment * lo, s.punishment * ~lo, s.contribution, s.others, s.P_others]
    b = la.lstsq(X, s.dc, rcond=None)[0]
    print(f"lag {k}: p(c<=10) {b[1]:.3f}  p(c>10) {b[2]:.3f}  c {b[3]:.3f}  others {b[4]:.3f}  others' p {b[5]:.3f}")
