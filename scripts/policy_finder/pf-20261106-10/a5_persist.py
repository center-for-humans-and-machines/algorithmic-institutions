# Persistence of the punishment effect: c_{t+k}-c_t on p_t, controlling for c_t and group mean, stayers.
import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-10')
import pandas as pd, numpy as np, statsmodels.formula.api as smf
from load import load
df=load()
df['gm']=df.groupby(['game','round_number','group_id']).contribution.transform('mean')
g=df.groupby(['game','player_id'])
for k in [1,2,3,4]:
    df[f'c{k}']=g.contribution.shift(-k); df[f'g{k}']=g.group_id.shift(-k)
d=df[(df.player_no_input==0)]
for k in [1,2,3,4]:
    e=d[(d[f'g{k}']==d.group_id)].copy(); e['y']=e[f'c{k}']-e.contribution
    e['low']=(e.contribution<e.gm-1).astype(float)
    m=smf.ols('y ~ punishment*low + contribution + gm', data=e).fit(cov_type='cluster',cov_kwds={'groups':e.game.astype('category').cat.codes})
    print(k, 'p:',m.params['punishment'].round(3),'p*low:',m.params['punishment:low'].round(3), 'se',m.bse['punishment'].round(3), 'n',len(e))
# switch-round timing: punishment in last round before decision vs earlier
s=df[df.round_number.isin([3,7,11,15,19])].copy(); s['sw']=(s.next_group!=s.group_id).astype(float)
for lag in [0,1,2,3]:
    s[f'p_l{lag}']=df.groupby(['game','player_id']).punishment.shift(lag).loc[s.index]
m=smf.logit('sw ~ p_l0+p_l1+p_l2+p_l3', data=s).fit(disp=0); print(m.params.round(3)); print(m.bse.round(3))
