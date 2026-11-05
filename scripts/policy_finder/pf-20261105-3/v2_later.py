"""v2: does punishment still work late in the game and on repeat offenders?
OLS per own-c group of the change k rounds later on own p (pieces), others' summed p,
others' mean c, for round windows; and split by whether the player was punished before."""
import numpy as np, pandas as pd
from v2_load import load
a = load(); c, p, v = a['c'], a['p'], a['v']
cum = np.concatenate([np.zeros_like(p[..., :1]), np.cumsum(p, -1)[..., :-1]], -1)  # punishment before t

def frame(t, k):
    m = v[..., t] & v[..., t + k]
    vt = v[..., t]; ps = p[..., t] * vt
    psum = ps.sum(-1, keepdims=True) - ps
    coth = ((c[..., t] * vt).sum(-1, keepdims=True) - c[..., t] * vt) / np.maximum(vt.sum(-1, keepdims=True) - vt, 1)
    return pd.DataFrame({'c': c[..., t][m], 'p': p[..., t][m], 'psum': np.broadcast_to(psum, m.shape)[m], 'coth': coth[m],
                         'cum': cum[..., t][m], 'd': (c[..., t + k] - c[..., t])[m]})

def fit(s):
    X = np.column_stack([np.ones(len(s)), s.c, s.coth, np.minimum(s.p, 3), np.clip(s.p - 3, 0, 5), np.clip(s.p - 8, 0, 30), s.psum])
    return np.round(np.linalg.lstsq(X, s.d.values, rcond=None)[0][3:], 3)

for (t0, t1, k) in [(1, 5, 1), (6, 11, 1), (12, 17, 1), (18, 22, 1), (12, 17, 5)]:
    D = pd.concat([frame(t, k) for t in range(t0, t1 + 1)])
    print(f'rounds {t0}-{t1}, k={k}: own p 0-3, 3-8, 8+ | others sum p')
    for lo, hi in [(0, 3), (4, 8), (9, 12), (13, 15), (16, 19)]:
        s = D[(D.c >= lo) & (D.c <= hi)]
        print(f'  c {lo:2d}-{hi:2d}: all {fit(s)}  never punished before {fit(s[s.cum == 0])}  punished >=10 before {fit(s[s.cum >= 10])}')
