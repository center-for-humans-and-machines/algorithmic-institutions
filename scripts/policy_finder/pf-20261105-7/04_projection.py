"""Local projections in the humanlike sim: c_{t+k} (own, others) on p_t, by own c bin."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
s = pd.read_parquet("plots/simulation/29_EXP1_reference/per_round.parquet")
s = s[s.run.str.contains("humanlike")].copy()
s.loc[~s.contribution_valid,"contribution"]=0
s = s.sort_values(["episode","participant_code","round_number"])
s["pid"] = s.participant_code.str.split("_").str[0].astype(int)
C = s.pivot_table(index=["episode","round_number"],columns="pid",values="contribution")
P = s.pivot_table(index=["episode","round_number"],columns="pid",values="punishment")
rows=[]
for ep in range(1000):
    c = C.loc[ep].values; p = P.loc[ep].values
    for t in range(24):
        for i in range(4):
            o=[j for j in range(4) if j!=i]
            r=dict(t=t,c=c[t,i],co=c[t,o].mean(),p=p[t,i],po=p[t,o].sum())
            if t>0: r.update(cl=c[t-1,i], pl=p[t-1,i])
            for k in range(1,7):
                if t+k<24: r[f"y{k}"]=c[t+k,i]; r[f"yo{k}"]=c[t+k,o].mean()
            rows.append(r)
d = pd.DataFrame(rows); d=d[d.t>0]
d["low"]=(d.c<=9).astype(int); d["mid"]=((d.c>9)&(d.c<20)).astype(int); d["full"]=(d.c==20).astype(int)
for k in range(1,7):
    m = smf.ols(f"y{k} ~ c + cl + co + pl + p:low + p:mid + p:full + po + t", d.dropna(subset=[f'y{k}'])).fit()
    mo = smf.ols(f"yo{k} ~ c + cl + co + pl + p:low + p:mid + p:full + po + t", d.dropna(subset=[f'y{k}'])).fit()
    print(k, "own:", m.params[["p:low","p:mid","p:full","po"]].round(3).to_dict(), " others-mean:", mo.params[["p:low","p:mid","p:full"]].round(3).to_dict())
# nonlinearity for low contributors
dl = d[d.c<=9].copy(); dl["pb"]=pd.cut(dl.p,[-1,0,3,6,9,12,16,20,30])
print(dl.groupby("pb",observed=True).apply(lambda x: pd.Series({"n":len(x),"y1-c":(x.y1-x.c).mean(),"y3-c":(x.y3-x.c).mean(),"y6-c":(x.y6-x.c).mean()})).round(2))
