"""v5: sweep seeds are per slot index (s000..s255) and shared across versions. How much of a point's score is its slot?"""
import json, numpy as np, pandas as pd
V = ['pf-20261112-5', 'pf-20261112-5-v2', 'pf-20261112-5-v3', 'pf-20261112-5-v4']
rows = []
for v in V:
    d = json.load(open(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/{v}_sweep/sweep.json'))
    for p in d['points']:
        k = int(p['name'].rsplit('_s', 1)[1])
        rows.append(dict(v=v, slot=k, pool=p['pool'], se=p['pool_se'], own=p['own'], other=p['other'], mem=p['members'], **{'P_' + a: b for a, b in p['params'].items()}))
df = pd.DataFrame(rows)
df.to_csv('/tmp/v5_points.csv', index=False)
W = df.pivot(index='slot', columns='v', values='pool')
print(W.corr().round(2))
print('mean per version', W.mean().round(2).to_dict(), 'sd', W.std().round(2).to_dict())
slot_mean = W.mean(axis=1)
print('sd of slot means %.2f; within-slot sd %.2f' % (slot_mean.std(), (W.sub(slot_mean, axis=0)).stack().std()))
