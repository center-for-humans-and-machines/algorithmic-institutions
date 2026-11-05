# v7: per design point x opponent (v2-v6 sweeps): opening breadth (share hit) and depth (points per hit) in rounds 0 and 1, later dose; vs pool
import json, numpy as np, pandas as pd, pyarrow.parquet as pq
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
rows = []
allp = {}
VS = ['pf-20261106-5-v2', 'pf-20261106-5-v3', 'pf-20261106-5-v4', 'pf-20261106-5-v5', 'pf-20261106-5-v6']
for v in VS:
    s = json.load(open(base + v + '_sweep/sweep.json'))
    for p in s['points']:
        for o, x in p['pools'].items():
            allp[(p['name'], o)] = (x['pool'], x['members'])
    for part in [1, 2]:
        d = pq.read_table(base + f'{v}_sweep/per_round_p{part}of2.parquet', columns=['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run'],
                          filters=[('episode', '<', 32)], read_dictionary=['run']).to_pandas()
        cats = pd.Series(d.run.cat.categories)
        m = cats.str.extract(r'managed by (\S+)_vs_(\S+)$')
        seat = np.where(m[0].str.startswith('pf-'), 0, 1)
        info = pd.DataFrame({'seat': seat, 'pt': np.where(seat == 0, m[0], m[1]), 'opp': np.where(seat == 0, m[1], m[0])})
        codes = d.run.cat.codes.values
        d = d.drop(columns='run')
        d['seat'] = info.seat.values[codes]; d['code'] = codes
        d = d[d.group_id == d.seat]
        d['hit'] = (d.punishment > 0).astype(float)
        r = d.round_number
        d['ph'] = np.select([r == 0, r == 1, r < 4, r < 16], ['r0', 'r1', 'r23', 'mid'], 'end')
        a = d.groupby(['code', 'ph']).agg(p=('punishment', 'mean'), hit=('hit', 'mean'), c=('contribution', 'mean')).unstack('ph')
        a.columns = [f'{x}_{y}' for x, y in a.columns]
        a = a.reset_index()
        a['pt'] = info.pt.values[a.code]; a['opp'] = info.opp.values[a.code]
        rows.append(a.drop(columns='code'))
a = pd.concat(rows).groupby(['pt', 'opp']).mean().reset_index()
a['pool'] = [allp[(p, o)][0] for p, o in zip(a.pt, a.opp)]
a['members'] = [allp[(p, o)][1] for p, o in zip(a.pt, a.opp)]
a['v'] = a.pt.str.extract(r'(v\d)')[0].fillna('v1')
a.to_csv('/tmp/v7_breadth.csv', index=False)
a['open'] = a.p_r0 + a.p_r1
a['depth0'] = a.p_r0 / a.hit_r0.clip(lower=1e-3)
print(a[['p_r0', 'hit_r0', 'depth0', 'p_r1', 'hit_r1', 'p_r23', 'p_mid', 'p_end', 'pool', 'members']].describe().round(2))
# within opening-dose bins: breadth vs pool
a['ob'] = pd.cut(a.open, [0, 4, 8, 12, 16, 20, 30, 60])
a['hb'] = pd.cut(a.hit_r0, [0, .2, .3, .4, .5, .6, 1])
print(a.pivot_table(index='ob', columns='hb', values='pool', aggfunc='mean', observed=True).round(1))
print(a.pivot_table(index='ob', columns='hb', values='pool', aggfunc='size', observed=True))
import statsmodels.formula.api as smf
a['o2'] = a.open ** 2
f = smf.ols('pool ~ C(opp) + open + o2 + hit_r0 + hit_r1 + p_r1 + p_r23 + I(p_r23**2) + p_mid + I(p_mid**2) + p_end', data=a).fit()
print(f.summary().tables[1])
