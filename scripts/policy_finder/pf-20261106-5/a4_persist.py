import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-5')
from load import load
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
d = load()
g = d.groupby(['game','player_id'])
for k in [1,2,3,4]:
    d[f'c{k}'] = g.contribution.shift(-k); d[f'g{k}'] = g.group_id.shift(-k)
x = d[(d.contribution <= 6) & (d.player_no_input == 0)]
for k in [1,2,3,4]:
    y = x[x[f'g{k}'] == x.group_id]
    y = y.assign(dc=y[f'c{k}'] - y.contribution)
    m = smf.ols('dc ~ contribution + gmean_c + punishment + gmean_p + round_number', y).fit()
    print(k, len(y), 'pun coef %.3f se %.3f' % (m.params.punishment, m.bse.punishment), 'gmean_p %.3f' % m.params.gmean_p)
# group-episode level: within each game, compare the two groups (manager effect)
G = d.groupby(['game','group_id','round_number']).agg(n=('player_id','size'), c=('contribution','mean'), p=('punishment','mean'), pool=('common_good','first')).reset_index()
E = G.groupby(['game','group_id']).agg(n=('n','mean'), c=('c','mean'), p=('p','mean'), pool=('pool','sum')).reset_index()
E['pool'] = E.pool / 24  # rounds the group existed; missing rounds = empty
print(E[['n','c','p','pool']].corr().round(2))
# within-game difference
W = E.pivot(index='game', columns='group_id')
dd = pd.DataFrame({'dp': W.p[0]-W.p[1], 'dn': W.n[0]-W.n[1], 'dc': W.c[0]-W.c[1], 'dpool': W.pool[0]-W.pool[1]})
print(dd.corr().round(2))
print(smf.ols('dpool ~ dp', dd).fit().params)
print(smf.ols('dn ~ dp', dd).fit().params)
# early punishment (rounds 0-3) vs later size
e = d[d.round_number<4].groupby(['game','group_id']).punishment.mean().unstack()
late = G[G.round_number>=4].groupby(['game','group_id']).n.mean().unstack().reindex(e.index).fillna(0)
print('corr early dp vs later dn', np.corrcoef(e[0]-e[1], late[0]-late[1])[0,1])
print(smf.ols('y ~ x', pd.DataFrame({'x': e[0]-e[1], 'y': late[0]-late[1]})).fit().params)
