"""v5: paired d by the race state at the first fine (from v5_first_state.py)."""
import pandas as pd, numpy as np
X = pd.read_parquet('/tmp/v5_first.parquet')
X['blk'] = X.tf // 4
def show(by, x=X):
    g = x.groupby(by, observed=True).agg(n=('d', 'size'), d=('d', 'mean'), se=('d', 'sem'), spend=('spend', 'mean'), dn=('dn', 'mean'))
    g['per_pt'] = g.d / (g.spend / 24)
    print(g.round(3).to_string()); print()
X['gb'] = pd.cut(X.pcgap, [-100, -10, -5, -2, 2, 5, 10, 100])
X['gbb'] = pd.cut(X.pcgap_blk, [-100, -10, -5, -2, 2, 5, 10, 100])
show('gb'); show('gbb'); show(['gbb'], X[X.blk >= 1])
X['ah_empty'] = X.n_ah == 0
show('ah_empty')
X['sb'] = pd.cut(X.short, [-30, 0, 4, 8, 12, 20])
show(['sb', pd.cut(X.pcgap_blk, [-100, 0, 5, 100])], X)
show(['blk', pd.cut(X.pcgap_blk, [-100, 0, 5, 100])], X)
print('=== large leads, finer')
Y = X[X.pcgap_blk > 5].copy()
Y['gl'] = pd.cut(Y.pcgap_blk, [5, 10, 15, 20, 30, 100])
show('gl', Y)
show(['gl', pd.cut(Y.short, [-30, 4, 8, 30])], Y)
show(['gl', Y.blk.clip(0, 3)], Y)
show(['gl', Y.n_me.clip(1, 6)], Y)
show(['gl', 'ver'], Y)
