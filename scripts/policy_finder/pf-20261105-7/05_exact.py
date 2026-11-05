"""Humanlike sim: next contribution by exact own c (few levels) x punishment bin; and
by whether anyone else was punished (spillover), holding own c and group mean."""
import pandas as pd, numpy as np
s = pd.read_parquet("plots/simulation/29_EXP1_reference/per_round.parquet")
s.loc[~s.contribution_valid,"contribution"]=0
s = s.sort_values(["run","episode","participant_code","round_number"])
g = s.groupby(["run","episode","participant_code"])
s["y1"]=g.contribution.shift(-1); s["v1"]=g.contribution_valid.shift(-1)
s["y3"]=g.contribution.shift(-3)
k=["run","episode","round_number"]
s["psum"]=s.groupby(k).punishment.transform("sum"); s["po"]=s.psum-s.punishment
s["co"]=(s.groupby(k).contribution.transform("sum")-s.contribution)/3
d=s[s.contribution_valid&(s.v1==True)].copy()
h=d[d.run.str.contains("humanlike")].copy()
h["pb"]=pd.cut(h.punishment,[-1,0,4,9,15,30])
for cv in [0,5,10,15,20]:
    x=h[(h.contribution==cv)]
    print("c=",cv, x.groupby("pb",observed=True).agg(n=("y1","size"),y1=("y1","mean"),y3=("y3","mean"),co=("co","mean")).round(2).to_dict("index"))
# spillover: own p=0, c>=10; others punished or not, binned by co
x=h[(h.punishment==0)&(h.contribution>=10)].copy()
x["cob"]=pd.cut(x.co,[0,5,10,15,20]); x["pob"]=pd.cut(x.po,[-1,0,5,15,90])
print((x.assign(dy=x.y1-x.contribution).pivot_table(index="cob",columns="pob",values="dy",observed=True)).round(2))
# same in zero sim for the reference (po=0 always)
z=d[d.run.str.contains("zero")&(d.contribution>=10)].copy(); z["cob"]=pd.cut(z.co,[0,5,10,15,20])
print(z.assign(dy=z.y1-z.contribution).groupby("cob",observed=True).dy.mean().round(2))
