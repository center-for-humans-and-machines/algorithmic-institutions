# Do finals share seeds across runs/versions? Compare round-0 contributions by episode.
import pandas as pd, numpy as np
base='plots/simulation/policy_finder/'
dfs=[]
for v in ['pf-20261105-7-v3','pf-20261105-7-v4','pf-20261105-7-v5','pf-20261105-7-v6']:
    dfs.append(pd.read_parquet(base+v+'_sweep/final/per_round.parquet'))
df=pd.concat(dfs)
df['run']=df.run.str.replace('ah exp1 managed by ','')
r0=df[df.round_number==0].copy()
r0['pi']=r0.participant_code.str.split('_').str[1].astype(int)
piv=r0.pivot_table(index=['episode','pi'],columns='run',values='contribution')
print(piv.head(8))
runs=piv.columns
print('corr of round-0 contributions between runs:'); print(piv.corr().round(2).iloc[:6,:6])
# episode-level score
df['valid']=df.contribution_valid.astype(float)
g=df[df.contribution_valid].groupby(['run','episode','round_number']).payoff.mean().groupby(['run','episode']).sum().unstack(0)
print(g.mean().round(2))
ref='pf-20261105-7-v3_gp'
for r in g.columns:
    d=g[r]-g[ref]; print(f'{r:28s} {g[r].mean():7.2f} diff vs v3_gp {d.mean():6.2f} se {d.std()/len(d)**.5:.2f} corr {np.corrcoef(g[r],g[ref])[0,1]:.2f}')
