import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['pair']=d.run.str.split('by ').str[1]
d=d.sort_values(['pair','episode','participant_code','round_number'])
k=['pair','episode','participant_code']
d['c_next']=d.groupby(k).contribution.shift(-1)
d['g_next']=d.groupby(k).agent_group.shift(-1)
d['dc']=d.c_next-d.contribution
d['mgr']=np.where(d.agent_group==0,d.pair.str.split('_vs_').str[0],d.pair.str.split('_vs_').str[1])
a=d[(d.mgr=='ah')&(d.g_next==d.agent_group)&(d.contribution_valid)]
a=a.assign(cbin=pd.cut(a.contribution,[-1,0,5,10,15,19,20]),pbin=pd.cut(a.punishment,[-1,0,2,5,10,30]))
print('AH-managed: dc by c x p'); print(a.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean',observed=True).round(2))
print(a.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='size',observed=True))
# persistence: contribution k rounds later after punishment, within same window & group
for lag in [1,2,3]:
    d[f'c{lag}']=d.groupby(k).contribution.shift(-lag)
    d[f'g{lag}']=d.groupby(k).agent_group.shift(-lag)
a=d[(d.mgr=='ah')&(d.contribution_valid)&(d.g3==d.agent_group)&(d.round_number%4==0)]
a=a.assign(cbin=pd.cut(a.contribution,[-1,0,5,10,15,20]),pbin=pd.cut(a.punishment,[-1,0,2,5,10,30]))
for lag in [1,2,3]:
    print('lag',lag); print(a.pivot_table(index='cbin',columns='pbin',values=f'c{lag}',aggfunc='mean',observed=True).sub(a.groupby('cbin',observed=True).contribution.mean(),axis=0).round(2))
# AH manager policy
b=d[d.mgr=='ah'].assign(cbin=pd.cut(d.contribution,[-1,0,5,10,15,19,20]))
print(b.groupby('cbin',observed=True).punishment.agg(['mean',lambda x:(x>0).mean()]).round(2))
print(b.groupby('round_number').punishment.mean().round(2).to_dict())
