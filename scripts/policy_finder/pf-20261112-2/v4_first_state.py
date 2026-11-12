"""v4: guard against post-treatment selection: the state at a game's FIRST fine is identical in
both arms. Paired d by my size / ah's size / round at the first fine, and d per point of the
game's spend (x24: -1 = pure cost)."""
import pandas as pd, numpy as np
P = pd.read_parquet('/tmp/v4_pairs.parquet')
E = pd.read_parquet('/tmp/v4_events2.parquet')
key = ['ver', 'idx', 'seat', 'episode']
f = E.sort_values('round_number').groupby(key).first()[['round_number', 'n_me', 'n_ah', 'contribution', 'gap']]
X = P.set_index(key).join(f, how='inner')
X['r'] = X.d / (X.spend / 24)
X['nme'] = X.n_me.clip(2, 7)
X['blk'] = X.round_number // 4
def show(by, x=X):
    g = x.groupby(by, observed=True).agg(d=('d', 'mean'), se=('d', 'sem'), spend=('spend', 'mean'), n=('d', 'size'))
    g['per_pt'] = g.d / (g.spend / 24)
    print(g.round(3).to_string())
show('nme')
show('nme', X[X.blk >= 1])
show(['blk'])
show(['blk', 'nme'], X[(X.blk >= 1)])
show('gap', X[X.blk >= 1])
X['band'] = pd.cut(X.contribution, [-1, 0, 4, 9, 20])
show('band', X[X.blk >= 1])
X['nah'] = X.n_ah.clip(0, 4)
show('nah', X[X.blk >= 1])
show('ver')
print('=== by gap at first fine, finer')
X['oth'] = None
f2 = E.sort_values('round_number').groupby(key).first()[['oth']]
X = X.drop(columns='oth').join(f2)
X['gapv'] = X.oth - X.contribution
X['gb'] = pd.cut(X.gapv, [-30, 0, 4, 8, 10, 12, 14, 20])
show('gb', X)
show(['gb', 'blk'], X[X.gapv >= 8])
show(['gb', 'ver'], X[X.gapv >= 8])
