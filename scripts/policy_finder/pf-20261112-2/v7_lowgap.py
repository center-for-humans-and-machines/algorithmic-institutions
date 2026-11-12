"""v7: v6's low-gap corner (gap < 5.5): coverage of the paired baseline, gain by fine, lead, n_min,
and an alternative estimate that treats uncovered fired games as 0 gain (lower-bound check)."""
import pandas as pd, numpy as np
P = pd.read_parquet('/tmp/v7_pts.parquet')
v6 = P[(P.ver == 'pf-20261112-2-v6')].copy()
v6['d_cons'] = v6.d_f.fillna(0) * v6.nf.fillna(0) * v6['cov'].fillna(0) / 500
def b(x):
    return pd.Series({'n': len(x), 'd_all': x.d_all.mean(), 'se': np.sqrt((x.se_all ** 2).sum()) / len(x), 'd_cons': x.d_cons.mean(),
                      'cov': x['cov'].mean(), 'fired': x.fired.mean(), 'spend': x.spend.mean(), 'dn_f': x.dn_f.mean()})
lo = v6[v6.gap < 5.5]
print('all v6', b(v6).round(3).to_dict()); print('gap<5.5', b(lo).round(3).to_dict())
for c, bins in [('fine', [1, 2, 3, 4, 5, 6]), ('lead', [0, 3, 6, 9, 12, 15]), ('n_min', [1, 2, 3, 4, 5]), ('last_block', [2, 3, 4, 5, 6])]:
    print('== gap<5.5 by', c); print(lo.groupby(pd.cut(lo[c], bins), observed=True).apply(b, include_groups=False).round(3).to_string())
x = lo[lo.fine > 4]
print('gap<5.5 & fine>4', b(x).round(3).to_dict())
print('gap<5.5 & fine>4 & n_min 3-4', b(x[x.n_min.isin([3, 4])]).round(3).to_dict())
# spend by block, low gap high fine
print(x[[f'sp{i}' for i in range(6)]].mean().round(2).to_dict())
