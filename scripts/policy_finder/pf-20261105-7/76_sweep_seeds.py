# Do sweep points share seeds? correlation of round-0 contributions / episode scores between points.
import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/policy_finder/pf-20261105-7-v6_sweep/per_round_p1of2.parquet')
d['run']=d.run.str.replace('ah exp1 managed by ','')
runs=d.run.unique()[:6]; print(runs, d.episode.max())
s=d[d.run.isin(runs)&d.contribution_valid].groupby(['run','episode','round_number']).payoff.mean().groupby(['run','episode']).sum().unstack(0)
print(s.corr().round(2))
r0=d[d.run.isin(runs)&(d.round_number==0)].groupby(['run','episode']).contribution.mean().unstack(0)
print(r0.corr().round(2))
