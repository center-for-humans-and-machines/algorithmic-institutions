"""v3 sweep: binned mean lead (and own/other/members) by parameter combos."""
import json, numpy as np, pandas as pd
sw = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261112-5-v3_sweep/sweep.json'))
rows = [dict(**p['params'], pool=p['pool'], own=p['own'], other=p['other'], mem=p['members']) for p in sw['points']]
d = pd.DataFrame(rows)
d['Db'] = pd.cut(d.D, [2, 4, 8, 14, 24.01])
d['hib'] = pd.cut(d.hi, [1, 4, 6, 8, 12])
f = lambda g: pd.Series(dict(n=len(g), pool=g.pool.mean(), se=g.pool.std() / np.sqrt(len(g)), own=g.own.mean(), other=g.other.mean(), mem=g.mem.mean()))
for keys in [['first'], ['first', 'hib'], ['first', 'Db'], ['first', 'nblocks'], ['first', 'w']]:
    print(d.groupby(keys, observed=True).apply(f).round(2))
s = d[(d['first'] == 0) & (d.hi <= 6)]
print('first=0 hi<=6'); print(s.groupby(['Db', 'nblocks'], observed=True).apply(f).round(2))
print(d.sort_values('pool', ascending=False).head(15).round(2))
