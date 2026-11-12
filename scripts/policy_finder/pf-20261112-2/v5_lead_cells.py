"""v5: cumulative cells 'first fine taken out of a per-capita lead >= L, target short >= S',
paired d (per game and per point of spend), and by fine size / block / version."""
import pandas as pd, numpy as np
X = pd.read_parquet('/tmp/v5_first.parquet')
X['blk'] = X.tf // 4
def row(x):
    return pd.Series({'n': len(x), 'd': x.d.mean(), 'se': x.d.sem(), 'spend': x.spend.mean(), 'per_pt': x.d.mean() / (x.spend.mean() / 24), 'dn': x.dn.mean(), 'npts': x.groupby(['ver', 'idx']).ngroups})
rows = []
for L in [0, 5, 8, 10, 12, 15, 20]:
    for S in [-30, 4, 8, 10, 12]:
        x = X[(X.pcgap_blk >= L) & (X.short >= S)]
        rows.append(pd.Series({'L': L, 'S': S}).append(row(x)) if hasattr(pd.Series, 'append') else pd.concat([pd.Series({'L': L, 'S': S}), row(x)]))
print(pd.DataFrame(rows).round(3).to_string())
x = X[(X.pcgap_blk >= 10) & (X.short >= 8)]
print(x.groupby(pd.cut(x.fine1, [0, 1.5, 3, 5, 8, 30])).apply(row).round(3))
print(x.groupby('blk').apply(row).round(3))
print(x.groupby('ver').apply(row).round(3))
print(x.groupby(pd.cut(x.tf % 4, [-1, 0, 1, 2, 3])).apply(row).round(3))
print('same cell, current-round gap instead of block gap')
x2 = X[(X.pcgap >= 10) & (X.short >= 8)]
print(row(x2).round(3))
print('=== lead >= 10, short >= 8: ah empty or not; blocks 0-1 only')
x = X[(X.pcgap_blk >= 10) & (X.short >= 8)]
print(x.groupby(x.n_ah == 0).apply(row).round(3))
x01 = x[x.blk <= 1]
print(x01.groupby(x01.n_ah == 0).apply(row).round(3))
for L in [5, 8, 10, 12, 15]:
    for S in [6, 8, 10]:
        y = X[(X.pcgap_blk >= L) & (X.short >= S) & (X.blk <= 1) & (X.n_ah > 0)]
        print(L, S, row(y).round(3).to_dict())
print('=== ah empty at the first fine')
z = X[X.n_ah == 0]
print(z.groupby(pd.cut(z.short, [-30, 0, 4, 8, 12, 20])).apply(row).round(3))
print(z.groupby('blk').apply(row).round(3))
print(z.groupby(pd.cut(z.fine1, [0, 1.5, 3, 6, 30])).apply(row).round(3))
