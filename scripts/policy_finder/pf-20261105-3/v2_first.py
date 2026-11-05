"""v2: is the first-ever punishment special, once the player's own past is controlled?
Rounds 3-17, change next round (and 4 later) by exact small p, per c group, for players
never punished before, with OLS controls c, c_{t-1}, own past mean, others' mean c, others' sum p."""
import numpy as np, pandas as pd
from v2_load import load
a = load(); c, p, v = a['c'], a['p'], a['v']
cum = np.concatenate([np.zeros_like(p[..., :1]), np.cumsum(p, -1)[..., :-1]], -1)
csum = np.cumsum(c * v, -1); vsum = np.cumsum(v, -1)

def frame(t, k):
    m = v[..., t] & v[..., t + k] & v[..., t - 1]
    vt = v[..., t]; ps = p[..., t] * vt
    psum = ps.sum(-1, keepdims=True) - ps
    coth = ((c[..., t] * vt).sum(-1, keepdims=True) - c[..., t] * vt) / np.maximum(vt.sum(-1, keepdims=True) - vt, 1)
    past = csum[..., t - 1] / np.maximum(vsum[..., t - 1], 1)
    return pd.DataFrame({'c': c[..., t][m], 'cl': c[..., t - 1][m], 'past': past[m], 'p': p[..., t][m],
                         'psum': np.broadcast_to(psum, m.shape)[m], 'coth': coth[m], 'cum': cum[..., t][m],
                         'd': (c[..., t + k] - c[..., t])[m]})

for k in [1, 4]:
    D = pd.concat([frame(t, k) for t in range(3, 18)])
    for name, s0 in [('never punished', D[D.cum == 0]), ('punished before (cum 1-15)', D[(D.cum > 0) & (D.cum <= 15)]), ('cum > 15', D[D.cum > 15])]:
        print(f'k={k} {name}: per c group, coef on dummies p=1,2,3,4-5,6-8,9-14,15+ (vs p=0), controls c, c_lag, past mean, coth, psum')
        for lo, hi in [(0, 3), (4, 8), (9, 12), (13, 15), (16, 19)]:
            s = s0[(s0.c >= lo) & (s0.c <= hi)]
            bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 14), (15, 30)]
            dums = np.column_stack([((s.p >= a_) & (s.p <= b_)).astype(float) for a_, b_ in bins])
            X = np.column_stack([np.ones(len(s)), s.c, s.cl, s.past, s.coth, s.psum, dums])
            ok = dums.sum(0) >= 150
            X = X[:, list(range(6)) + [6 + i for i in range(len(bins)) if ok[i]]]
            b = np.linalg.lstsq(X, s.d.values, rcond=None)[0]
            out = ['   .' for _ in bins]; j = 6
            for i in range(len(bins)):
                if ok[i]: out[i] = f'{b[j]:5.2f}'; j += 1
            print(f'  c {lo:2d}-{hi:2d} n={len(s):7d} lag {b[2]:.2f} past {b[3]:.2f} psum {b[5]:.3f} |', ' '.join(out))
