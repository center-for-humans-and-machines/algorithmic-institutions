"""Reference sims: contribution response to punishment and leaving vs punishment, in the AH stack."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('managed by ah_vs_ah|managed by ah_vs_zero|managed by zero_vs_ah')]
d = d.sort_values(['run','episode','participant_code','round_number'])
g = d.groupby(['run','episode','participant_code'])
d['c_next'] = g.contribution.shift(-1); d['grp_next'] = g.group_id.shift(-1)
v = d[(d.contribution_valid==1)]
w = v[(v.grp_next==v.group_id)&v.c_next.notna()].copy()
w['dc'] = w.c_next-w.contribution
w['cbin']=pd.cut(w.contribution,[-1,0,4,9,14,19,20]); w['pbin']=pd.cut(w.punishment,[-1,0,2,5,10,30])
print(w.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean').round(2))
print(w.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='count'))
# persistence: contribution k rounds later vs punishment at t, for c in 0-9, same group throughout
for k in [1,2,3,4]:
    d[f'c{k}'] = g.contribution.shift(-k); d[f'g{k}'] = g.group_id.shift(-k)
x = d[(d.contribution_valid==1)&(d.contribution<=9)]
for k in [1,2,3,4]:
    y = x[x[f'g{k}']==x.group_id]
    print(k, y.groupby(pd.cut(y.punishment,[-1,0,2,5,10,30]))[f'c{k}'].mean().sub(y.groupby(pd.cut(y.punishment,[-1,0,2,5,10,30])).contribution.mean()).round(2).to_dict())
# leaving in sims
d['block']=d.round_number//4
pl = d.groupby(['run','episode','participant_code','block']).agg(grp=('group_id','last'),p=('punishment','sum'),p_last=('punishment','last'),c=('contribution','mean')).reset_index()
pl['gn']=pl.groupby(['run','episode','participant_code']).grp.shift(-1); pl=pl[pl.gn.notna()]; pl['left']=(pl.gn!=pl.grp).astype(int)
print('\nsim leave rate by block punishment'); print(pl.groupby(pd.cut(pl.p,[-1,0,3,8,15,30,200])).left.agg(['mean','count']).round(3))
print('sim leave by last-round punishment'); print(pl.groupby(pd.cut(pl.p_last,[-1,0,3,8,30])).left.agg(['mean','count']).round(3))
import statsmodels.formula.api as smf
pl['pe']=pl.p-pl.p_last
r=smf.logit('left ~ p_last + pe + c', data=pl[pl.run.str.contains('ah_vs_ah')]).fit(disp=0); print(r.params.round(4))
