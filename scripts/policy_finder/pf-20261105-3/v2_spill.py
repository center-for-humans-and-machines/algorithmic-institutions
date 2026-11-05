"""v2: shape of the spillover in round 0 -> 1 (and -> 5): change of an unpunished or
lightly punished player's contribution by others' total punishment, sum vs max."""
import numpy as np, pandas as pd
from v2_load import load
a = load(); c, p, v = a['c'], a['p'], a['v']
t = 0
for k in [1, 5]:
    m = v[..., t] & v[..., t + k]
    vt = v[..., t]
    ps = p[..., t] * vt
    psum = ps.sum(-1, keepdims=True) - ps
    pmax = np.stack([np.max(np.delete(ps, i, axis=-1), -1) for i in range(4)], -1)
    coth = ((c[..., t] * vt).sum(-1, keepdims=True) - c[..., t] * vt) / np.maximum(vt.sum(-1, keepdims=True) - vt, 1)
    df = pd.DataFrame({'c': c[..., t][m], 'p': p[..., t][m], 'psum': np.broadcast_to(psum, c[..., t].shape)[m],
                       'pmax': pmax[m], 'coth': coth[m], 'd': (c[..., t + k] - c[..., t])[m]})
    hi = df[(df.c >= 15) & (df.p == 0)].copy()
    hi['sb'] = pd.cut(hi.psum, [-1, 0, 5, 10, 20, 30, 45, 60, 90])
    hi['cob'] = pd.cut(hi.coth, [-1, 7, 12, 20])
    print(f'k={k}: c>=15 unpunished, change by others sum p (rows) x others mean c (cols)')
    print(hi.pivot_table(index='sb', columns='cob', values='d', aggfunc='mean', observed=True).round(2).to_string())
    X = np.column_stack([np.ones(len(hi)), hi.c, hi.coth, np.minimum(hi.psum, 10), np.clip(hi.psum - 10, 0, 20), np.clip(hi.psum - 30, 0, 90), hi.pmax])
    print('coef psum 0-10, 10-30, 30+, pmax:', np.round(np.linalg.lstsq(X, hi.d.values, rcond=None)[0][3:], 3))
    lo = df[(df.c <= 8)].copy()
    X = np.column_stack([np.ones(len(lo)), lo.c, lo.coth, np.minimum(lo.p, 3), np.clip(lo.p - 3, 0, 5), np.clip(lo.p - 8, 0, 30), np.minimum(lo.psum, 10), np.clip(lo.psum - 10, 0, 20), np.clip(lo.psum - 30, 0, 90)])
    print('c<=8: own 0-3,3-8,8+ | others sum 0-10,10-30,30+:', np.round(np.linalg.lstsq(X, lo.d.values, rcond=None)[0][3:], 3))
