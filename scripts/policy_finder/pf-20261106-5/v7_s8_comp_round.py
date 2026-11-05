# v7: like v7_s6/s7 for another round r: my group's count giving <= 8 in round r vs how many were hit in round r -> pool from round r on
# usage: v7_s8_comp_round.py r
import sys, json, numpy as np, pandas as pd, pyarrow.parquet as pq, statsmodels.formula.api as smf
R = int(sys.argv[1])
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
rows = []
for v in ['pf-20261106-5-v4', 'pf-20261106-5-v5', 'pf-20261106-5-v6']:
    for part in [1, 2]:
        d = pq.read_table(base + f'{v}_sweep/per_round_p{part}of2.parquet', columns=['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run'],
                          filters=[('episode', '<', 32), ('round_number', '>=', R)], read_dictionary=['run']).to_pandas()
        cats = pd.Series(d.run.cat.categories)
        m = cats.str.extract(r'managed by (\S+)_vs_(\S+)$')
        seat = np.where(m[0].str.startswith('pf-'), 0, 1)
        opp = np.where(seat == 0, m[1], m[0])
        codes = d.run.cat.codes.values
        d = d.drop(columns='run'); d['code'] = codes
        d = d[d.group_id.values == seat[codes]]
        g = d.groupby(['code', 'episode'])
        tot = g.agg(c=('contribution', 'sum'), p=('punishment', 'sum'))
        tot['pool'] = (1.6 * tot.c - tot.p) / (24 - R)
        rr = d[d.round_number == R]
        rr = rr.assign(low=(rr.contribution <= 8).astype(int), hit=(rr.punishment > 0).astype(int))
        r0 = rr.groupby(['code', 'episode']).agg(nlow=('low', 'sum'), n=('low', 'size'), nhit=('hit', 'sum'), pr=('punishment', 'sum'), cr=('contribution', 'sum'))
        x = tot.join(r0, how='inner').reset_index()
        x['opp'] = opp[x.code]; x['v'] = v
        rows.append(x.drop(columns='code'))
a = pd.concat(rows)
a['depth'] = a.pr / a.nhit.clip(lower=1)
a = a[((a.nhit == 0) | (a.depth >= 8))]
for n in sorted(a.n.unique()):
    if n < 3 or n > 6: continue
    for nl in range(1, n):
        s = a[(a.n == n) & (a.nlow == nl)]
        if len(s) < 500: continue
        s = s.assign(k=s.nhit.astype(str))
        ks = s.k.value_counts()
        s = s[s.k.isin(ks[ks >= 100].index)]
        if str(nl) not in set(s.k): continue
        f = smf.ols('pool ~ C(k, Treatment("%d")) + cr + C(opp) + C(v)' % nl, data=s).fit()
        co = {k.split('[T.')[1][:-1]: (round(f.params[k], 1), round(f.bse[k], 1)) for k in f.params.index if 'C(k' in k}
        print('round', R, 'size', n, 'nlow', nl, 'ref k=%d' % nl, co, 'n', len(s), dict(ks))
