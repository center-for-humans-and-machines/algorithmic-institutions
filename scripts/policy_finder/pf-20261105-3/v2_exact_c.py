"""v2: round-0 response by exact own contribution (round 0 has no history, so the
rule's punishment is exogenous given c); next-round change by c x p bin, plus
regression with others' punishment and contributions as controls."""
import numpy as np, pandas as pd
from v2_load import load
a = load(); c, p, v = a['c'], a['p'], a['v']

def frame(t, k=1):
    m = v[..., t] & v[..., t + k]
    vt = v[..., t]
    n_oth = vt.sum(-1, keepdims=True) - vt
    poth = ((p[..., t] * vt).sum(-1, keepdims=True) - p[..., t] * vt) / np.maximum(n_oth, 1)
    coth = ((c[..., t] * vt).sum(-1, keepdims=True) - c[..., t] * vt) / np.maximum(n_oth, 1)
    return pd.DataFrame({'c': c[..., t][m], 'p': p[..., t][m], 'poth': poth[m], 'coth': coth[m],
                         'd': (c[..., t + k] - c[..., t])[m], 't': t})

df = frame(0)
df['pb'] = pd.cut(df.p, [-1, 0, 2, 4, 7, 10, 15, 30])
tab = df.pivot_table(index='c', columns='pb', values='d', aggfunc='mean', observed=True).round(1)
n = df.pivot_table(index='c', columns='pb', values='d', aggfunc='size', observed=True)
print('round 0 -> 1 change by exact c x p bin'); print(tab.where(n >= 100).to_string())
# regression per c-group with piecewise own p and others p
def pw(x, knots):
    return np.stack([np.clip(x - lo, 0, hi - lo) for lo, hi in zip(knots[:-1], knots[1:])], 1)
for t0, t1, k in [(0, 0, 1), (0, 0, 5), (1, 10, 1), (1, 10, 5)]:
    D = pd.concat([frame(t, k) for t in range(t0, t1 + 1)])
    print(f'\nrounds {t0}-{t1}, change after {k}: coefs (per point) own p pieces 0-3,3-8,8-15,15-30 | poth | coth')
    for lo, hi in [(0, 3), (4, 8), (9, 12), (13, 15), (16, 17), (18, 19), (20, 20)]:
        s = D[(D.c >= lo) & (D.c <= hi)]
        X = np.column_stack([np.ones(len(s)), s.c, pw(s.p.values, [0, 3, 8, 15, 30]), s.poth, s.coth])
        b = np.linalg.lstsq(X, s.d.values, rcond=None)[0]
        print(f'c {lo:2d}-{hi:2d} n={len(s):7d}', np.round(b[2:], 3))
