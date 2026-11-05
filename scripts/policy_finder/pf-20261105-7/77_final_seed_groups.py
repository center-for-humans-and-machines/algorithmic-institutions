# Which final runs share seeds? correlation of episode scores among all final runs of v1-v6.
import pandas as pd, numpy as np
base='plots/simulation/policy_finder/'
vs=['pf-20261105-7','pf-20261105-7-v2','pf-20261105-7-v3','pf-20261105-7-v4','pf-20261105-7-v5','pf-20261105-7-v6']
S={}
for v in vs:
    d=pd.read_parquet(base+v+'_sweep/final/per_round.parquet')
    d['run']=d.run.str.replace('ah exp1 managed by pf-20261105-7','')
    s=d[d.contribution_valid].groupby(['run','episode','round_number']).payoff.mean().groupby(['run','episode']).sum().unstack(0)
    for r in d.run.unique(): S[r]=s[r]
S=pd.DataFrame(S)
C=S.corr()
for r in C:
    hi=[f'{q}({C.loc[r,q]:.2f})' for q in C if q!=r and C.loc[r,q]>0.3]
    print(f'{r:12s} {S[r].mean():7.2f} paired with: {" ".join(hi)}')
