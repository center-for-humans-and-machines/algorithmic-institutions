"""v4: value of a point of fine by category, from the paired games: regress the per-game paired
difference d on the game's spend per category / 24 (no intercept; -1 = pure cost, 0 = pays for
itself). SE by bootstrap over the 101 point pairs."""
import pandas as pd, numpy as np, sys
P = pd.read_parquet('/tmp/v4_pairs.parquet')
E = pd.read_parquet('/tmp/v4_events.parquet')
E['period'] = pd.cut(E.round_number, [-1, 3, 19, 23], labels=['r0-3', 'r4-19', 'r20-23']).astype(str)
E['band'] = pd.cut(E.contribution, [-1, 0, 4, 9, 20], labels=['c0', 'c1-4', 'c5-9', 'c10+']).astype(str)
E['phase'] = 'ph' + (E.round_number % 4).astype(str)
E['nme'] = 'nme' + E.n_me.clip(2, 6).astype(str)
E['nah'] = 'nah' + E.n_ah.clip(1, 5).astype(str)
E['oth'] = (E.cs_me - E.contribution) / (E.n_me - 1).clip(lower=1)
E['gap'] = pd.cut(E.oth - E.contribution, [-30, 2, 6, 10, 30], labels=['g<2', 'g2-6', 'g6-10', 'g10+']).astype(str)
key = ['ver', 'idx', 'seat', 'episode']
def fit(cat, sub=None):
    e = E if sub is None else E[sub(E)]
    rest = E.drop(e.index)
    S = e.pivot_table(index=key, columns=cat, values='punishment', aggfunc='sum', fill_value=0) / 24
    if len(rest):
        S['other'] = rest.groupby(key).punishment.sum().reindex(S.index).fillna(0) / 24
    X = P.set_index(key)[['d']].join(S).fillna(0)
    cols = [c for c in X.columns if c != 'd']
    codes = pd.factorize(X.index.droplevel([2, 3]))[0]
    upairs = np.unique(codes)
    grp = {p: np.where(codes == p)[0] for p in upairs}
    A = X[cols].values; y = X.d.values
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    rng = np.random.default_rng(0); bs = []
    for _ in range(200):
        ii = np.concatenate([grp[upairs[k]] for k in rng.integers(0, len(upairs), len(upairs))])
        bs.append(np.linalg.lstsq(A[ii], y[ii], rcond=None)[0])
    se = np.std(bs, axis=0)
    tot = A.sum(axis=0) * 24 / len(X)
    print(f'--- {cat}' + ('' if sub is None else ' (subset)'))
    for c, bb, s, t in zip(cols, b, se, tot):
        print(f'{c:>14s} {bb:7.2f} +- {s:5.2f}   spend/game {t:6.2f}')
E['pb'] = E.period + '_' + E.band
E.to_parquet('/tmp/v4_events2.parquet')
fit('pb')
mid = lambda e: e.period == 'r4-19'
E['phb'] = E.phase + '_' + np.where(E.band == 'c0', 'c0', np.where(E.band == 'c10+', 'c10+', 'c1-9'))
fit('phb', mid)
fit('nme', mid)
fit('nah', mid)
fit('gap', mid)
E['blk'] = 'b' + (E.round_number // 4).astype(str)
fit('blk')
E['vv'] = E.ver; fit('vv')

