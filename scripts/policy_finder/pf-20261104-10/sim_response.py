"""Reference sims (artificial humans): contribution change vs punishment and leaving vs punishment."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('ah_vs_ah|ah_vs_zero|zero_vs_ah')]
d = d.sort_values(['run','episode','participant_code','round_number'])
g = d.groupby(['run','episode','participant_code'])
d['c_next'] = g.contribution.shift(-1); d['grp_next'] = g.agent_group.shift(-1)
d['v_next'] = g.contribution_valid.shift(-1)
d['dc'] = d.c_next - d.contribution
mgr = d.apply(lambda r: None, axis=1) if False else None
pair = d.run.str.split('managed by ').str[-1].str.split('_vs_')
d['mgr'] = np.where(d.agent_group==0, pair.str[0], pair.str[1])
v = d[(d.contribution_valid==1)&(d.v_next==1)&(d.grp_next==d.agent_group)&(~d.round_number.isin([3,7,11,15,19,23]))&(d.mgr=='ah')]
v['pbin']=pd.cut(v.punishment,[-1,0,3,6,10,30]); v['cbin']=pd.cut(v.contribution,[-1,4,9,14,19,20])
print('AH-contrib: dc by own c (rows) x punishment (cols), ah-managed groups')
print(v.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean',observed=True).round(2))
print(v.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='count',observed=True))
# zero-managed baseline
z = d[(d.contribution_valid==1)&(d.v_next==1)&(d.grp_next==d.agent_group)&(~d.round_number.isin([3,7,11,15,19,23]))&(d.mgr=='zero')]
z['cbin']=pd.cut(z.contribution,[-1,4,9,14,19,20])
print('zero-managed dc by c', z.groupby('cbin',observed=True).dc.mean().round(2).values)
# leaving
d['block']=d.round_number//4
b = d[d.block<5].groupby(['run','episode','participant_code','block']).agg(grp=('agent_group','last'),
    p=('punishment','mean'), c=('contribution','mean'), gn=('grp_next','last'), mgr=('mgr','last')).reset_index()
b['left']=(b.gn!=b.grp).astype(float)
b['pbin']=pd.cut(b.p,[-1,0,1,3,6,30])
print('leave by punishment in block, by manager'); print(b.pivot_table(index='pbin',columns='mgr',values='left',aggfunc=['mean','count'],observed=True).round(3))
print('leave by block'); print(b.pivot_table(index='block',columns='mgr',values='left',aggfunc='mean').round(3))
b['cbin']=pd.cut(b.c,[-1,5,10,15,20])
print(b[b.mgr=='zero'].groupby('cbin',observed=True).left.mean().round(3))
