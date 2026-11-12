"""Switching at decision rounds vs punishment received at lags within the block, and vs group-level quantities."""
import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d=d.sort_values(['run','episode','participant_code','round_number'])
k=['run','episode','participant_code']
d['g_next']=d.groupby(k).agent_group.shift(-1)
d['switch']=(d.g_next!=d.agent_group).astype(float)
for L in range(4): d[f'p_l{L}']=d.groupby(k).punishment.shift(L)
for L in range(4): d[f'c_l{L}']=d.groupby(k).contribution.shift(L)
d['mgr']=np.where(d.agent_group==0, d.run.str.split('managed by ').str[-1].str.split('_vs_').str[0], d.run.str.split('_vs_').str[-1])
# group-level: own group pool per cap vs other group's
g=d.groupby(['run','episode','round_number','agent_group']).agg(gp=('punishment','mean'),gc=('contribution','mean'),n=('contribution','size'),pay=('payoff','mean')).reset_index()
o=g.copy(); o['agent_group']=1-o.agent_group; o=o.rename(columns={'gp':'ogp','gc':'ogc','n':'on','pay':'opay'})
d=d.merge(g,on=['run','episode','round_number','agent_group']).merge(o,on=['run','episode','round_number','agent_group'],how='left')
s=d[d.round_number.isin([7,11,15,19])&d.g_next.notna()&(d.mgr=='ah')].copy()
import numpy.linalg as la
cols=['p_l0','p_l1','p_l2','p_l3','c_l0','gp','gc','n','ogc','on']
s=s.dropna(subset=cols)
X=np.column_stack([np.ones(len(s))]+[s[c].values for c in cols]); b=la.lstsq(X,s.switch.values,rcond=None)[0]
print('LPM switch (ah-managed, rounds 7-19):'); [print(f'  {c:6s} {v:+.4f}') for c,v in zip(['const']+cols,b)]
# whole-sample: zero-managed own player: does the other group's punishment level pull?
s2=d[d.round_number.isin([7,11,15,19])&d.g_next.notna()].copy().dropna(subset=['ogp'])
s2['payd']=s2.pay-s2.opay
s2['pdb']=pd.cut(s2.payd,[-50,-5,-2,0,2,5,50])
print(s2.pivot_table(index='mgr',columns='pdb',values='switch').round(3))
print('own-pay minus other-group mean pay, binned:')
s2['ppd']=s2.payoff-s2.opay; s2['ppdb']=pd.cut(s2.ppd,[-50,-10,-5,-2,0,2,5,50])
print(s2.pivot_table(index='mgr',columns='ppdb',values='switch').round(3))
