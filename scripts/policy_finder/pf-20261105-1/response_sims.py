import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
print(d.columns.tolist(), d.shape)
d['mgr'] = d.run.str.split().str[-1]
d['valid'] = d.contribution_valid.astype(int)
d.loc[d.valid==0,'contribution'] = 0
d = d.sort_values(['mgr','episode','participant_code','round_number'])
g = d.groupby(['mgr','episode','participant_code'])
for k in [1,2,3,5]:
    d[f'c{k}'] = g.contribution.shift(-k); d[f'v{k}'] = g.valid.shift(-k)
print('timeout rate', d.groupby('mgr').valid.apply(lambda s:1-s.mean()))
# score
vv = d[d.valid==1]
pr = vv.groupby(['mgr','episode','round_number']).payoff.mean().groupby(['mgr','episode']).sum()
print('score (sum over rounds of mean valid payoff):', pr.groupby('mgr').agg(['mean','sem']))
h = d[(d.mgr=='humanlike')&(d.valid==1)&(d.v1==1)].copy()
h['dc'] = h.c1-h.contribution
h['pbin'] = pd.cut(h.punishment,[-1,0,2,5,10,30]); h['cbin']=pd.cut(h.contribution,[-1,4,9,14,19,20])
print('\nhumanlike sim: dc next round by cbin x pbin'); print(h.pivot_table(index='cbin',columns='pbin',values='dc',observed=False).round(2))
print(h.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='count',observed=False))
print('\nhumanlike punishment by own c bin', h.groupby('cbin',observed=False).punishment.agg(['mean','count']).round(2))
for k in [2,3,5]:
    x = d[(d.mgr=='humanlike')&(d.valid==1)&(d[f'v{k}']==1)].copy()
    x['dc']=x[f'c{k}']-x.contribution
    x['pbin'] = pd.cut(x.punishment,[-1,0,2,5,10,30]); x['cbin']=pd.cut(x.contribution,[-1,4,9,14,19,20])
    print(f'\nhumanlike dc at +{k}'); print(x.pivot_table(index='cbin',columns='pbin',values='dc',observed=False).round(2))
z = d[(d.mgr=='zero')&(d.valid==1)&(d.v1==1)].copy(); z['dc']=z.c1-z.contribution
z['cbin']=pd.cut(z.contribution,[-1,4,9,14,19,20])
print('\nzero dc by cbin', z.groupby('cbin',observed=False).dc.agg(['mean','count']).round(2))
print('\nzero c dist round 0', d[(d.mgr=='zero')&(d.round_number==0)].contribution.describe())
