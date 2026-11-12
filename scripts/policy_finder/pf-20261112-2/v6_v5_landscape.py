"""v6: v5's sweep points scored by the per-game baseline (v6_gamebase), d_all = fired share * mean
paired effect of the covered fired games; binned and regressed on v5's params."""
import pandas as pd, numpy as np, json
pt = pd.read_csv('scripts/policy_finder/pf-20261112-2/v6_gamebase_points.csv')
v5 = json.load(open('plots/simulation/policy_finder/pf-20261112-2-v5_sweep/sweep.json'))
par = pd.DataFrame([dict(idx=int(p['name'].split('_s')[-1]), **p['params']) for p in v5['points']])
s = pt[pt.ver == 'pf-20261112-2-v5'].merge(par, on='idx')
s['se_all'] = s.se_fired.fillna(0) * s.fired
on = s[s.fine >= 1]
print('v5 fining points', len(on), 'mean d_all %.3f +- %.3f' % (on.d_all.mean(), np.sqrt((on.se_all ** 2).sum()) / len(on)))
def b(x):
    return pd.Series({'n': len(x), 'd_all': x.d_all.mean(), 'se': np.sqrt((x.se_all ** 2).sum()) / len(x),
                      'd_fired': x.d_fired.mean(), 'fired': x.fired.mean(), 'spend': x.spend.mean(), 'cov': x['cov'].mean()})
for c, bins in [('lead', [0, 5, 10, 15, 20, 25]), ('gap', [5, 7, 9, 11, 13, 15]), ('fine', [1, 2, 3, 4, 5, 6]), ('last_block', [0, 1, 2, 3, 4, 5, 6])]:
    print(c); print(on.groupby(pd.cut(on[c], bins), observed=True).apply(b, include_groups=False).round(3).to_string())
print('lead x gap'); print(on.groupby([pd.cut(on.lead, [0, 8, 16, 25]), pd.cut(on.gap, [5, 8, 11, 15])], observed=True).apply(b, include_groups=False).round(3).to_string())
print('gap x last_block'); print(on.groupby([pd.cut(on.gap, [5, 8, 11, 15]), pd.cut(on.last_block, [0, 3, 6])], observed=True).apply(b, include_groups=False).round(3).to_string())
X = np.c_[np.ones(len(on)), on.lead, on.gap, on.fine, on.last_block]
print('OLS d_all ~ 1 + lead + gap + fine + last_block', np.linalg.lstsq(X, on.d_all, rcond=None)[0].round(4))
print(on.sort_values('d_all').tail(15)[['idx', 'lead', 'gap', 'fine', 'last_block', 'fired', 'cov', 'd_fired', 'd_all', 'se_all', 'spend']].round(2).to_string())
