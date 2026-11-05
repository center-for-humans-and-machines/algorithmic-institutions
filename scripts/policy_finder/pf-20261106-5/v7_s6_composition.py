# v7: episode level (v4-v6 sweeps): my group's round-0 composition (how many of its 4 gave <= 8) x how many were hit in round 0 -> episode pool
import json, numpy as np, pandas as pd, pyarrow.parquet as pq
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
rows = []
for v in ['pf-20261106-5-v4', 'pf-20261106-5-v5', 'pf-20261106-5-v6']:
    for part in [1, 2]:
        d = pq.read_table(base + f'{v}_sweep/per_round_p{part}of2.parquet', columns=['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run'],
                          filters=[('episode', '<', 32)], read_dictionary=['run']).to_pandas()
        cats = pd.Series(d.run.cat.categories)
        m = cats.str.extract(r'managed by (\S+)_vs_(\S+)$')
        seat = np.where(m[0].str.startswith('pf-'), 0, 1)
        opp = np.where(seat == 0, m[1], m[0])
        codes = d.run.cat.codes.values
        d = d.drop(columns='run'); d['code'] = codes
        d['mine'] = d.group_id.values == seat[codes]
        d = d[d.mine]
        g = d.groupby(['code', 'episode'])
        tot = g.agg(c=('contribution', 'sum'), p=('punishment', 'sum'))
        tot['pool'] = (1.6 * tot.c - tot.p) / 24
        r0 = d[d.round_number == 0].groupby(['code', 'episode']).agg(nlow=('contribution', lambda x: (x <= 8).sum()),
                n=('contribution', 'size'), nhit=('punishment', lambda x: (x > 0).sum()), p0=('punishment', 'sum'), c0=('contribution', 'sum'))
        x = tot.join(r0).reset_index()
        x['opp'] = opp[x.code]; x['v'] = v
        rows.append(x.drop(columns='code'))
a = pd.concat(rows)
a.to_csv('/tmp/v7_comp.csv', index=False)
a['pool_d'] = a.pool - a.groupby('opp').pool.transform('mean')
a['depth'] = a.p0 / a.nhit.clip(lower=1)
a = a[(a.nhit == 0) | (a.depth >= 12)]
print(a.pivot_table(index='nlow', columns='nhit', values='pool_d', aggfunc='mean').round(2))
print(a.pivot_table(index='nlow', columns='nhit', values='pool_d', aggfunc='size'))
print(a.pivot_table(index='nlow', columns='nhit', values='pool_d', aggfunc='sem').round(2))
