# v6: per design point and opponent: my opening dose (rounds 0-1) and later dose (rounds 2-23) vs pool, pooled over v3-v5 sweeps
import json, re, numpy as np, pandas as pd, pyarrow.parquet as pq
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
pd.set_option('display.width', 250)
rows = []
for v in ['pf-20261106-5-v3', 'pf-20261106-5-v4', 'pf-20261106-5-v5']:
    s = json.load(open(base + v + '_sweep/sweep.json'))
    pools = {(p['name'], o): x['pool'] for p in s['points'] for o, x in p['pools'].items()}
    for part in [1, 2]:
        d = pq.read_table(base + f'{v}_sweep/per_round_p{part}of2.parquet', columns=['episode', 'round_number', 'punishment', 'group_id', 'run'],
                          filters=[('episode', '<', 32)], read_dictionary=['run']).to_pandas()
        cats = pd.Series(d.run.cat.categories)
        m = cats.str.extract(r'managed by (\S+)_vs_(\S+)$')
        seat = np.where(m[0].str.startswith('pf-'), 0, 1)
        info = pd.DataFrame({'seat': seat, 'pt': np.where(seat == 0, m[0], m[1]), 'opp': np.where(seat == 0, m[1], m[0])})
        codes = d.run.cat.codes.values
        d = d.drop(columns='run')
        d['seat'] = info.seat.values[codes]; d['code'] = codes
        d = d[d.group_id == d.seat]
        d['pt'] = info.pt.values[d.code.values]; d['opp'] = info.opp.values[d.code.values]
        d['ph'] = np.where(d.round_number < 2, 'open', 'late')
        a = d.groupby(['pt', 'opp', 'ph']).punishment.mean().unstack('ph').reset_index()
        rows.append(a)
a = pd.concat(rows).groupby(['pt', 'opp']).mean().reset_index()
a['pool'] = [pools.get((p, o), np.nan) for p, o in zip(a.pt, a.opp)] if False else np.nan
allp = {}
for v in ['pf-20261106-5-v3', 'pf-20261106-5-v4', 'pf-20261106-5-v5']:
    s = json.load(open(base + v + '_sweep/sweep.json'))
    for p in s['points']:
        for o, x in p['pools'].items():
            allp[(p['name'], o)] = (x['pool'], x['members'])
a['pool'] = [allp[(p, o)][0] for p, o in zip(a.pt, a.opp)]
a['members'] = [allp[(p, o)][1] for p, o in zip(a.pt, a.opp)]
a['ob'] = pd.cut(a.open, [-0.1, 2, 4, 6, 8, 10, 13, 30])
a['lb'] = pd.cut(a.late, [-0.1, 0.25, 0.5, 1, 1.5, 2.5, 30])
print(a.pivot_table(index='ob', columns='opp', values='pool', aggfunc='mean', observed=True).round(1))
print(a.pivot_table(index='ob', columns='opp', values='members', aggfunc='mean', observed=True).round(2))
print(a.groupby('ob', observed=True).size())
print(a.pivot_table(index='lb', columns='opp', values='pool', aggfunc='mean', observed=True).round(1))
print(a.groupby('lb', observed=True).size())
a.to_csv('/tmp/v6_dose_opp.csv', index=False)
