import pandas as pd, sys
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['in']=1
print(d.groupby('run').episode.nunique())
# common_good per group-round: one value per group per round
g=d.groupby(['run','episode','round_number','group_id']).agg(pool=('common_good','first'),n=('in','sum'),c=('contribution','mean'),p=('punishment','mean')).reset_index()
print(g.groupby(['run','group_id'])[['pool','n','c','p']].mean())
# per round trajectories
for r in g.run.unique():
    x=g[g.run==r].groupby(['round_number','group_id'])[['pool','n','c','p']].mean().unstack()
    print(r); print(x.round(2).to_string())
