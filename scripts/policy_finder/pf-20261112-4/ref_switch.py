# switching in reference sims: who leaves after decision rounds 3,7,11,15,19
import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['c']=d.contribution.where(d.contribution_valid.astype(bool),0)
d=d.sort_values(['run','episode','participant_code','round_number'])
g=d.groupby(['run','episode','participant_code'])
d['next_group']=g.group_id.shift(-1)
d['next_c']=g.c.shift(-1)
d['sw']=(d.next_group!=d.group_id).astype(float)
pool=d.groupby(['run','episode','round_number','group_id']).apply(lambda x:1.6*x.c.sum()-x.punishment.sum(),include_groups=False).rename('pool').reset_index()
d=d.merge(pool,on=['run','episode','round_number','group_id'])
op=pool.copy(); op['group_id']=1-op.group_id; op=op.rename(columns={'pool':'opool'})
d=d.merge(op,on=['run','episode','round_number','group_id'],how='left'); d['opool']=d.opool.fillna(0)
dec=d[d.round_number.isin([3,7,11,15,19])].copy()
print('switch rate by run and decision round'); print(dec.groupby(['run','round_number']).sw.mean().unstack().round(3))
x=dec[dec.run.str.contains('ah_vs_ah')]
x['pb']=pd.cut(x.punishment,[-1,0,2,5,10,30])
print('ah_vs_ah: switch rate by own punishment at decision round'); print(x.groupby('pb').sw.agg(['mean','size']).round(3))
x['dpool']=pd.cut(x.pool-x.opool,[-200,-30,-10,0,10,30,200])
print('by own pool - other pool'); print(x.groupby('dpool').sw.agg(['mean','size']).round(3))
# punishment in the previous non-decision rounds
d['p_prev']=g.punishment.shift(1).values if False else d.groupby(['run','episode','participant_code']).punishment.shift(1)
dec=d[d.round_number.isin([3,7,11,15,19])&d.run.str.contains('ah_vs_ah')].copy()
dec['pp']=pd.cut(dec.p_prev,[-1,0,2,5,10,30])
print('ah_vs_ah: switch by punishment one round before decision'); print(dec.groupby('pp').sw.agg(['mean','size']).round(3))
# contribution response
y=d[d.run.str.contains('ah_vs_ah')&(d.round_number<23)&(d.sw==0)].copy()
y['pb']=pd.cut(y.punishment,[-1,0,2,5,10,30]); y['cb']=pd.cut(y.c,[-1,4,9,14,20])
print('ah_vs_ah: next c - c by punishment and contribution level'); print(y.assign(dc=y.next_c-y.c).groupby(['cb','pb']).dc.agg(['mean','size']).unstack(1).round(2))
