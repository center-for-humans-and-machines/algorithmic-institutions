import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-12')
from load import load
import numpy as np, pandas as pd, statsmodels.api as sm
d, gs = load()
d = d.sort_values(['game','player_id','round_number'])
g = d.groupby(['game','player_id'])
for k in range(4):
    d[f'p_l{k}'] = g.punishment.shift(k)
d['cg_pc'] = d.pool / d.n
dec = d[d.round_number.isin([3,7,11,15,19])].copy()
dec['sw'] = (dec.grp_next != dec.group_id).astype(float)
og = gs.copy(); og['group_id'] = 1-og.group_id; og['ocg_pc']=og.pool/og.n
dec = dec.merge(og[['game','round_number','group_id','ocg_pc']], on=['game','round_number','group_id'], how='left')
dec['ocg_pc']=dec.ocg_pc.fillna(0)
X = dec[['p_l0','p_l1','p_l2','p_l3','cg_pc','ocg_pc','contribution']].copy()
X[['cg_pc','ocg_pc']] /= 10
m = sm.Logit(dec.sw, sm.add_constant(X)).fit(disp=0); print(m.summary2().tables[1].round(3))
# does being punished when low contributor -> leave? and does leaving of low contributors help?
dec['low'] = dec.contribution < dec.cmean - 3
print(dec.groupby(['low', pd.cut(dec.p_l0,[-1,0,3,8,30])], observed=True).sw.agg(['mean','size']).round(3))
# who joins: contribution of switchers in the new group vs before
sw = d[(d.grp_next!=d.group_id)&d.grp_next.notna()]
print('switchers contribution at decision', sw.contribution.mean().round(2), 'next round', sw.c_next.mean().round(2),
      'stayers', d[(d.grp_next==d.group_id)&d.round_number.isin([3,7,11,15,19])].contribution.mean().round(2))
