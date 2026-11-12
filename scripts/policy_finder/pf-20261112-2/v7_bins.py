"""v7: v6's (and v5's) sweep points by their paired gain d_all (v7_points), binned per parameter and
in pairs, plus an OLS with quadratic terms, to see where the gain sits."""
import pandas as pd, numpy as np
P = pd.read_parquet('/tmp/v7_pts.parquet')
def b(x):
    return pd.Series({'n': len(x), 'd_all': x.d_all.mean(), 'se': np.sqrt((x.se_all ** 2).sum()) / len(x),
                      'd_f': x.d_f.mean(), 'fired': x.fired.mean(), 'spend': x.spend.mean()})
v6 = P[P.ver == 'pf-20261112-2-v6']
for c, bins in [('lead', [0, 2.5, 5, 7.5, 10, 12.5, 15]), ('gap', [3, 4, 5, 6, 7, 8, 10, 12]), ('fine', [1, 2, 3, 4, 5, 6]), ('last_block', [2, 3, 4, 5, 6]), ('n_min', [1, 2, 3, 4, 5])]:
    print('== v6 by', c); print(v6.groupby(pd.cut(v6[c], bins), observed=True).apply(b, include_groups=False).round(3).to_string())
print('== v6 n_min x last_block'); print(v6.groupby(['n_min', 'last_block']).apply(b, include_groups=False).round(3).to_string())
print('== v6 gap x lead'); print(v6.groupby([pd.cut(v6.gap, [3, 5, 7, 12]), pd.cut(v6.lead, [0, 5, 10, 15])], observed=True).apply(b, include_groups=False).round(3).to_string())
print('== v6 gap x fine'); print(v6.groupby([pd.cut(v6.gap, [3, 5, 7, 12]), pd.cut(v6.fine, [1, 2.5, 4, 6])], observed=True).apply(b, include_groups=False).round(3).to_string())
print('== v6 n_min x gap'); print(v6.groupby([v6.n_min, pd.cut(v6.gap, [3, 5, 7, 12])], observed=True).apply(b, include_groups=False).round(3).to_string())
A = P[P.fine >= 1].copy()
z = lambda s: (s - s.mean()) / s.std()
X = pd.DataFrame({'one': 1, 'lead': z(A.lead.clip(upper=15)), 'gap': z(A.gap), 'fine': z(A.fine), 'lb': z(A.last_block), 'nmin': z(A.n_min)})
for c in ['lead', 'gap', 'fine', 'lb', 'nmin']:
    X[c + '2'] = X[c] ** 2
w = 1 / (A.se_all ** 2 + 0.05 ** 2)
beta = np.linalg.lstsq(X.values * np.sqrt(w.values)[:, None], A.d_all.values * np.sqrt(w.values), rcond=None)[0]
print('WLS (v5+v6, standardised):', dict(zip(X.columns, beta.round(3))))
print('top v6 points'); print(v6.sort_values('d_all').tail(15)[['idx', 'lead', 'gap', 'fine', 'last_block', 'n_min', 'fired', 'spend', 'd_f', 'd_all', 'se_all']].round(2).to_string())
