"""v6: paired effect per fired game (v6_gamebase baseline) by the state at the game's first fine,
v4 and v5 sweeps: race (block-to-date per-capita lead), my size, the target's shortfall, round."""
import pandas as pd, numpy as np
X = pd.read_parquet('/tmp/v6_first.parquet')
def a(x):
    return pd.Series({'n': len(x), 'd': x.d.mean(), 'se': x.d.sem(), 'spend': x.spend.mean(),
                      'per_pt': x.d.sum() / x.spend.sum() * 24, 'dn': x.dn.mean()})
X['race'] = pd.cut(X.pcgap_blk, [-99, -10, -5, -2, 0, 2, 5, 10, 99])
X['size'] = X.n_me.round().clip(1, 8)
X['sh'] = pd.cut(X.short, [-99, 3, 5, 7, 9, 12, 99])
X['blk'] = (X.tf // 4).astype(int)
for v in ['pf-20261112-2-v4', 'pf-20261112-2-v5']:
    x = X[X.ver == v]
    print('=====', v, a(x).round(3).to_dict())
    for c in ['race', 'size', 'sh', 'blk']:
        print(x.groupby(c, observed=True).apply(a, include_groups=False).round(3).to_string())
x = X
print('== both: race x size')
print(x.groupby([pd.cut(x.pcgap_blk, [-99, 0, 5, 99]), pd.cut(x.n_me, [0, 3.5, 4.5, 9])], observed=True).apply(a, include_groups=False).round(3).to_string())
print('== both: race x short')
print(x.groupby([pd.cut(x.pcgap_blk, [-99, -5, 0, 5, 99]), x.sh], observed=True).apply(a, include_groups=False).round(3).to_string())
print('== both: fine size at first fine')
print(x.groupby(pd.cut(x.fine1, [0, 2, 4, 6, 8, 10]), observed=True).apply(a, include_groups=False).round(3).to_string())
print('== both: finer race lines, groups of 4+')
y = X[X.n_me >= 3.5]
print(y.groupby(pd.cut(y.pcgap_blk, [-99, 0, 2.5, 5, 7.5, 10, 15, 20, 99]), observed=True).apply(a, include_groups=False).round(3).to_string())
print('== race > 5, groups 4+: block x version')
z = y[y.pcgap_blk > 5]
print(z.groupby(['ver', 'blk']).apply(a, include_groups=False).round(3).to_string())
