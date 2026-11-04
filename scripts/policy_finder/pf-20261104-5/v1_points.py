"""v2: across v1's 256 sweep points, per point: punishment per member-round by block, members,
member contribution and pool by block (our group = 0). Saves a table for later looks."""
import pandas as pd, numpy as np, json, re
B = 'plots/simulation/policy_finder/pf-20261104-5_sweep/'
S = json.load(open(B + 'sweep.json'))
par = pd.DataFrame([dict(p['params'], name=p['name'], score=p['pool'], se=p['pool_se']) for p in S['points']])
out = []
for f in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    d = pd.read_parquet(B + f, columns=['run', 'episode', 'round_number', 'group_id', 'contribution', 'punishment'], filters=[('group_id', '==', 0)])
    d['run'] = d.run.astype('category')
    d['name'] = d.run.cat.rename_categories(lambda r: re.search(r'by (\S+)_vs', r).group(1))
    d['blk'] = np.minimum(d.round_number // 4, 3)  # 0-3, 4-7, 8-11, 12-23
    g = d.groupby(['name', 'episode', 'round_number', 'blk'], observed=True).agg(n=('contribution', 'size'), c=('contribution', 'sum'), p=('punishment', 'sum')).reset_index()
    g['pool'] = 1.6 * g.c - g.p
    a = g.groupby(['name', 'blk'], observed=True).agg(n=('n', 'mean'), c=('c', 'sum'), p=('p', 'sum'), nn=('n', 'sum'), pool=('pool', 'mean')).reset_index()
    a['cm'] = a.c / a.nn; a['pm'] = a.p / a.nn
    out.append(a.pivot(index='name', columns='blk', values=['n', 'cm', 'pm', 'pool']))
t = pd.concat(out)
t.columns = [f'{a}{b}' for a, b in t.columns]
t = par.set_index('name').join(t)
t.to_csv('scripts/policy_finder/pf-20261104-5/v1_points.csv')
print(t.describe().round(2).T)
