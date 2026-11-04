"""v1 sweep: per design point, own punishment by phase, members, contribution per member, pool;
then regress pool on the phase doses. Writes a csv cache."""
import json, os, pandas as pd, numpy as np
base = 'plots/simulation/policy_finder/pf-20261104-11_sweep/'
cache = 'scripts/policy_finder/pf-20261104-11/v1_sweep_features.csv'
if not os.path.exists(cache):
    rows = []
    for f in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
        import pyarrow.parquet as pq
        d = pq.read_table(base + f, columns=['episode', 'round_number', 'group_id', 'contribution', 'punishment', 'run'], read_dictionary=['run']).to_pandas()
        cats = d.run.cat.categories.str.extract(r'_s(\d+)_vs_ah')[0].astype(int).values
        d['pt'] = cats[d.run.cat.codes.values]
        d = d.drop(columns='run')
        d['ph'] = pd.cut(d.round_number, [-1, 0, 2, 3, 7, 11, 19, 23], labels=['r0', 'r1-2', 'r3', 'r4-7', 'r8-11', 'r12-19', 'r20-23'])
        own = d[d.group_id == 0]
        ne = d.groupby('pt').episode.nunique()
        a = own.groupby(['pt', 'ph'], observed=True).punishment.sum().unstack() / ne.values[:, None]
        nr = pd.Series({'r0': 1, 'r1-2': 2, 'r3': 1, 'r4-7': 4, 'r8-11': 4, 'r12-19': 8, 'r20-23': 4})
        a = (a / nr).add_prefix('P_')
        a['members'] = own.groupby('pt').size() / ne / 24
        a['m_r4'] = own[own.round_number == 4].groupby('pt').size() / ne
        a['c_late'] = own[own.round_number >= 8].groupby('pt').contribution.mean()
        a['c_r0_3'] = own[own.round_number < 4].groupby('pt').contribution.mean()
        a['frac_pun'] = (own.punishment > 0).groupby(own.pt).mean()
        a['ah_c_late'] = d[(d.group_id == 1) & (d.round_number >= 8)].groupby('pt').contribution.mean()
        rows.append(a)
    F = pd.concat(rows)
    sw = json.load(open(base + 'sweep.json'))['points']
    P = pd.DataFrame([{**p['params'], 'pool': p['pool'], 'pt': int(p['name'].split('_s')[-1])} for p in sw]).set_index('pt')
    F = F.join(P)
    F.to_csv(cache)
F = pd.read_csv(cache, index_col=0)
pd.set_option('display.width', 250)
print(F.sort_values('pool', ascending=False).round(2).head(25).to_string())
print(F.sort_values('pool').round(2).head(10).to_string())
print(F.corr()['pool'].round(2))
import numpy.linalg as la
X = F[[c for c in F.columns if c.startswith('P_')]].copy()
X['P_r0sq'] = X['P_r0']**2
X['P_r1-2sq'] = X['P_r1-2']**2
X['one'] = 1
b = la.lstsq(X.values, F.pool.values, rcond=None)[0]
print(dict(zip(X.columns, b.round(3))))
