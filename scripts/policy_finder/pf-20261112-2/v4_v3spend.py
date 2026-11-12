"""v4: v3's sweep, per design point: total spend per game, members, lead; regress lead on spend."""
import json, pandas as pd, numpy as np
x = pd.read_parquet('/tmp/v3_pr_pf-20261112-2-v3.parquet')
d = json.load(open('plots/simulation/policy_finder/pf-20261112-2-v3_sweep/sweep.json'))
P = pd.DataFrame([dict(pt=p['name'], **p['params'], pool=p['pool'], m=p['members'], own=p['own'], oth=p['other']) for p in d['points']])
s = x.groupby('pt').agg(spend=('p_me', 'sum')).reset_index()
s['spend'] /= 2  # two seats
s['spend_b0'] = x[x.round_number < 4].groupby('pt').p_me.sum().values / 2
P = P.merge(s, on='pt')
P['logsp'] = np.log1p(P.spend)
P['any'] = (P.spend > 0.5).astype(float)
print(P[['spend', 'spend_b0']].describe().round(1))
P['sb'] = pd.cut(P.spend, [-1, 0.5, 5, 15, 40, 100, 1000])
print(P.groupby('sb', observed=True)[['pool', 'own', 'oth', 'm']].agg(['mean']).round(2).assign(n=P.groupby('sb', observed=True).size()))
import numpy.linalg as la
X = np.column_stack([np.ones(len(P)), P['any'], P.spend])
b = la.lstsq(X, P.pool, rcond=None)[0]; print('pool ~ 1 + any + spend', b.round(3))
P.to_csv('/tmp/v4_v3pts.csv', index=False)
