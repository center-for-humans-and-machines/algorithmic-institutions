# v6: block-start (rounds 4, 8, 12) punishment dose vs rest-of-game pool, by my group's size then (v5 sweep, + v4)
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
pd.set_option('display.width', 250)
res = []
for v in sys.argv[1:]:
    for part in [1, 2]:
        d = pq.read_table(base + f'{v}_sweep/per_round_p{part}of2.parquet', columns=['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run'],
                          filters=[('episode', '<', 32), ('round_number', '>=', 4)], read_dictionary=['run']).to_pandas()
        cats = pd.Series(d.run.cat.categories)
        seat = np.where(cats.str.contains('managed by pf-'), 0, 1)
        codes = d.run.cat.codes.values
        d = d.drop(columns='run'); d['code'] = codes; d['seat'] = seat[codes]
        d = d[d.group_id == d.seat]
        g = d.groupby(['code', 'episode', 'round_number']).agg(n=('contribution', 'size'), c=('contribution', 'sum'), p=('punishment', 'sum')).reset_index()
        g['pool'] = 1.6 * g.c - g.p
        for r in [4, 8, 12]:
            x = g[g.round_number == r][['code', 'episode', 'n', 'c', 'p']].rename(columns={'n': 'n0', 'c': 'c0', 'p': 'p0'})
            fut = g[g.round_number >= r].groupby(['code', 'episode']).pool.sum().rename('fut').reset_index()
            n4 = g[g.round_number == r + 4][['code', 'episode', 'n']].rename(columns={'n': 'n4'})
            x = x.merge(fut, on=['code', 'episode']).merge(n4, on=['code', 'episode'], how='left')
            x['r'] = r; x['v'] = v + str(part)
            res.append(x)
x = pd.concat(res)
x['n4'] = x.n4.fillna(0)
x['mc'] = x.c0 / x.n0; x['dose'] = x.p0 / x.n0
x['db'] = pd.cut(x.dose, [-0.1, 0, 2, 5, 10, 30])
x['cb'] = pd.cut(x.mc, [0, 9, 11, 13, 20])
x['nb'] = x.n0.clip(1, 6)
for r in [4, 8, 12]:
    y = x[x.r == r]
    print('round', r)
    print(y.pivot_table(index=['nb'], columns='db', values='fut', aggfunc='mean', observed=True).round(1).to_string())
    print(y.pivot_table(index=['nb'], columns='db', values='n4', aggfunc='mean', observed=True).round(2).to_string())
    print(y.pivot_table(index=['nb'], columns='db', values='fut', aggfunc='size', observed=True).to_string())
# control for mean contribution: within nb x cb cells
y = x[x.r.isin([4, 8])]
print(y.pivot_table(index=['nb', 'cb'], columns='db', values='fut', aggfunc='mean', observed=True).round(1).to_string())
import statsmodels.formula.api as smf
x['cb2'] = pd.cut(x.mc, [0, 7, 9, 10.5, 12, 14, 20]).astype(str); x['db'] = x.db.astype(str)
for r in [4, 8, 12]:
    y = x[x.r == r].copy()
    m = smf.ols('fut ~ C(db) * C(cb2) + n0 + I(n0**2) + mc', data=y).fit()
    grid = pd.DataFrame([(db, cb) for db in sorted(y.db.unique()) for cb in sorted(y.cb2.unique())], columns=['db', 'cb2'])
    grid['n0'] = 4; grid['mc'] = y.mc.mean()
    grid['pred'] = m.predict(grid)
    print('round', r, '(pred. rest-of-game pool at 4 members, adj. for mean c)')
    print(grid.pivot(index='cb2', columns='db', values='pred').round(0).to_string())
    print(y.pivot_table(index='cb2', columns='db', values='fut', aggfunc='size', observed=True).to_string())
