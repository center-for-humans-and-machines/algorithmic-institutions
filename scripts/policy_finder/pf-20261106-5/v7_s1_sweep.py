# v7: v6 sweep, finals and landscape, per opponent
import json, sys, numpy as np, pandas as pd
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
for v in ['pf-20261106-5', 'pf-20261106-5-v2', 'pf-20261106-5-v3', 'pf-20261106-5-v4', 'pf-20261106-5-v5', 'pf-20261106-5-v6']:
    f = json.load(open(base + v + '_sweep/final/sweep.json'))
    print('FINAL', v, json.dumps({k: f[k] for k in f if k != 'points'}, default=str)[:400])
    for p in f['points']:
        print(' ', p['name'], {k: round(float(x), 2) for k, x in p['params'].items()}, round(p['pool'], 2), round(p['pool_se'], 2),
              {k: (round(x['pool'], 1), round(x['members'], 2)) for k, x in p['pools'].items()})
s = json.load(open(base + 'pf-20261106-5-v6_sweep/sweep.json'))
rows = []
for p in s['points']:
    r = dict(p['params']); r['name'] = p['name']; r['pool'] = p['pool']; r['members'] = p['members']
    for k, x in p['pools'].items():
        r[k] = x['pool']; r[k + '_m'] = x['members']
    rows.append(r)
d = pd.DataFrame(rows)
print(d.sort_values('pool', ascending=False).head(20).round(2))
print(d.describe().round(2))
d['lslope'] = np.log(d.slope)
P = ['target_open', 'lslope', 'target_late', 'size_gain', 'window_late']
for o in ['pool'] + s['opponents']:
    print(o, d[P].corrwith(d[o]).round(2).to_dict())
for c in P:
    d['b'] = pd.qcut(d[c], 5, duplicates='drop')
    print(d.groupby('b', observed=True)[['pool'] + s['opponents']].mean().round(1))
L = json.load(open(base + 'pf-20261106-5-v6_sweep/landscapes.json'))
print(json.dumps({k: L[k] for k in L if k not in ('points', 'profiles')}, indent=0, default=str)[:2500])
for k, pr in L.get('profiles', {}).items():
    print(k, str(pr)[:600])
d.to_csv('/tmp/v6sweep.csv', index=False)
