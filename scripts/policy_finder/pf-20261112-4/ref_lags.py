# sim switch model: does punishment early in a block weigh less than at the decision round?
import pandas as pd, numpy as np, statsmodels.formula.api as smf
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d=d[d.run.str.contains('ah_vs_ah|ah_vs_zero')]
d['c']=d.contribution.where(d.contribution_valid.astype(bool),0)
d=d.sort_values(['run','episode','participant_code','round_number'])
g=d.groupby(['run','episode','participant_code'])
d['sw']=(g.group_id.shift(-1)!=d.group_id).astype(float)
for k in [1,2,3,4]: d[f'p{k}']=g.punishment.shift(k)
k=['run','episode','round_number','group_id']
pool=d.groupby(k).apply(lambda x:1.6*x.c.sum()-x.punishment.sum(),include_groups=False).rename('pool').reset_index()
op=pool.copy(); op['group_id']=1-op.group_id; op=op.rename(columns={'pool':'opool'})
d=d.merge(pool,on=k).merge(op,on=k,how='left'); d['opool']=d.opool.fillna(0)
dec=d[d.round_number.isin([7,11,15,19])&((d.run.str.contains('ah_vs_ah'))|((d.run.str.contains('ah_vs_zero'))&(d.group_id==0)))].copy()
dec['dpool']=(dec.pool-dec.opool)/10
m=smf.logit('sw ~ punishment + p1 + p2 + p3 + p4 + c + dpool + C(round_number)',data=dec).fit(disp=0)
print(m.params.round(3)); print(m.bse.round(3))
