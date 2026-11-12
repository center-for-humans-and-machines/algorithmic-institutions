"""Reference sims: how next-round contribution and switching respond to punishment."""
import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d=d.sort_values(['run','episode','participant_code','round_number'])
k=['run','episode','participant_code']
d['c_next']=d.groupby(k).contribution.shift(-1)
d['g_next']=d.groupby(k).agent_group.shift(-1)
d['valid_next']=d.groupby(k).contribution_valid.shift(-1)
d['switch']=(d.g_next!=d.agent_group).astype(float)
# group mean contribution this round
d['gc']=d.groupby(['run','episode','round_number','agent_group']).contribution.transform('mean')
d['gn']=d.groupby(['run','episode','round_number','agent_group']).contribution.transform('size')
ah=d[d.punishment.notna()]
# which rows are ah-managed: groups with any punishment? use run name
def mgr(r):
    g0,g1=r.run.split('managed by ')[-1].split('_vs_'); return g0 if r.agent_group==0 else g1
d['mgr']=np.where(d.agent_group==0, d.run.str.split('managed by ').str[-1].str.split('_vs_').str[0], d.run.str.split('_vs_').str[-1])
x=d[(d.mgr=='ah')&d.c_next.notna()&(d.g_next==d.agent_group)&(d.contribution_valid==1)&(d.valid_next==1)].copy()
x['dc']=x.c_next-x.contribution
x['pb']=pd.cut(x.punishment,[-1,0,2,5,10,30])
x['cb']=pd.cut(x.contribution,[-1,0,5,10,15,20])
print('ah-managed, staying, valid: mean dc by contribution bin x punishment bin')
print(x.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean').round(2))
print(x.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size'))
# ah punishment policy
y=d[(d.mgr=='ah')&(d.contribution_valid==1)]
y['cb']=pd.cut(y.contribution,[-1,0,5,10,15,20])
print('ah mean punishment by contribution bin', y.groupby('cb').punishment.agg(['mean','size']).round(2))
# zero managed
z=d[(d.mgr=='zero')&d.c_next.notna()&(d.g_next==d.agent_group)&(d.contribution_valid==1)&(d.valid_next==1)].copy()
z['dc']=z.c_next-z.contribution; z['cb']=pd.cut(z.contribution,[-1,0,5,10,15,20])
print('zero-managed dc by contrib bin', z.groupby('cb').dc.mean().round(2))
# switching at decision rounds
s=d[d.round_number.isin([3,7,11,15,19])&d.g_next.notna()].copy()
s['pb']=pd.cut(s.punishment,[-1,0,2,5,10,30])
print('switch prob by mgr and punishment bin (this round)')
print(s.pivot_table(index='mgr',columns='pb',values='switch',aggfunc='mean').round(3))
print(s.pivot_table(index='mgr',columns='pb',values='switch',aggfunc='size'))
print('switch prob by mgr and round'); print(s.pivot_table(index='mgr',columns='round_number',values='switch').round(3))
