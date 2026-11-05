# v2: how did v1's sweep scores depend on params, per opponent?
import json, numpy as np, pandas as pd
s = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-5_sweep/sweep.json'))
rows = []
for p in s['points']:
    r = dict(p['params']); r['name'] = p['name']; r['pool'] = p['pool']; r['members'] = p['members']
    for k, v in p['pools'].items():
        r[k] = v['pool']; r[k + '_m'] = v['members']
    rows.append(r)
d = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(d.sort_values('pool', ascending=False).head(25).round(2))
print(d.sort_values('pool').head(10).round(2))
print(d.describe().round(2))
d['lslope'] = np.log(d.slope)
for o in ['pool'] + s['opponents']:
    print(o, d[['lslope', 'tol', 'cap0', 'follow', 't_stop']].corrwith(d[o]).round(2).to_dict())
# rough "punishment strength" proxy
d['strength'] = d.slope * np.maximum(0, 6 - d.tol)
d['sb'] = pd.qcut(d.strength, 6)
print(d.groupby('sb')[['pool'] + s['opponents']].mean().round(1))
d['tb'] = pd.cut(d.t_stop, [7, 10, 13, 16, 19, 23])
print(d[d.strength > d.strength.median()].groupby('tb')[['pool'] + s['opponents']].mean().round(1))
d.to_csv('/tmp/v1sweep.csv', index=False)
