"""Per-player response: c_{t+1} - c_t vs punishment p_t, by own c_t; pilots and reference sims."""
import pandas as pd, numpy as np
df = pd.read_csv("experiments/pilot_random1_player_round_slim.csv")
df = df.sort_values(["global_group_id","episode","participant_code","round_number"])
g = df.groupby(["global_group_id","episode","participant_code"])
df["c_next"] = g.contribution.shift(-1); df["valid_next"] = 1-g.player_no_input.shift(-1)
grp = df.groupby(["global_group_id","episode","round_number"])
df["c_others"] = (grp.contribution.transform("sum")-df.contribution)/3
d = df[(df.player_no_input==0)&(df.valid_next==1)].copy()
d["dc"] = d.c_next-d.contribution
d["cbin"] = pd.cut(d.contribution,[-1,4,9,14,19,20])
d["pbin"] = pd.cut(d.punishment,[-1,0,3,7,12,30])
print("PILOTS mean dc by own c (rows) and punishment (cols)")
print(d.pivot_table(index="cbin",columns="pbin",values="dc",aggfunc="mean",observed=True).round(2))
print(d.pivot_table(index="cbin",columns="pbin",values="dc",aggfunc="count",observed=True))
import statsmodels.formula.api as smf
m = smf.ols("c_next ~ contribution + c_others + punishment + I(punishment**2) + I(punishment*(contribution==20))", d).fit()
print(m.params.round(3), m.bse.round(3))
# reference sims
s = pd.read_parquet("plots/simulation/29_EXP1_reference/per_round.parquet")
s = s.sort_values(["run","episode","participant_code","round_number"])
s.loc[~s.contribution_valid,"contribution"]=0
gs = s.groupby(["run","episode","participant_code"])
s["c_next"]=gs.contribution.shift(-1); s["v_next"]=gs.contribution_valid.shift(-1)
s["c_others"]=(s.groupby(["run","episode","round_number"]).contribution.transform("sum")-s.contribution)/3
e = s[s.contribution_valid & (s.v_next==True)].copy()
e["dc"]=e.c_next-e.contribution
e["cbin"]=pd.cut(e.contribution,[-1,4,9,14,19,20]); e["pbin"]=pd.cut(e.punishment,[-1,0,3,7,12,30])
for run, x in e.groupby("run"):
    print("SIM", run)
    print(x.pivot_table(index="cbin",columns="pbin",values="dc",aggfunc="mean",observed=True).round(2))
    print(x.pivot_table(index="cbin",columns="pbin",values="dc",aggfunc="count",observed=True))
h = e[e.run.str.contains("humanlike")]
m2 = smf.ols("c_next ~ contribution + c_others + punishment + I(punishment**2)", h).fit()
print(m2.params.round(3))
