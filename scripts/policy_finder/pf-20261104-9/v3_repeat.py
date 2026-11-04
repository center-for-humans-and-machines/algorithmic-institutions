"""Does a second punishment add as much as the first? v3 sweep, my group, members still laggards in
round 1 (c1 / others' mean in [lo, hi)). OLS of c at round 11 and of staying after round 3 on round-1
punishment p1 (linear and sqrt), split by round-0 punishment p0 bins; controls c1, c0, others' mean, po1."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/v3.npz'); C = z['C'].astype(float); V = z['V']; P = z['P'].astype(float); G = z['G']
C = C.reshape(-1, 8, 24); V = V.reshape(-1, 8, 24); P = P.reshape(-1, 8, 24); G = G.reshape(-1, 8, 24)
def ctx(t):
    m = (G[:, :, t] == 0) * V[:, :, t]; k = m.sum(1, keepdims=True)
    o = ((C[:, :, t] * m).sum(1, keepdims=True) - C[:, :, t] * m) / np.maximum(k - m, 1)
    po = ((P[:, :, t] * m).sum(1, keepdims=True) - P[:, :, t] * m) / np.maximum(k - m, 1)
    return m, k, np.broadcast_to(o, m.shape), np.broadcast_to(po, m.shape)
m0, k0, o0, _ = ctx(0); m1, k1, o1, po1 = ctx(1)
sel = (m0 == 1) & (m1 == 1) & (k1 >= 2)
df = pd.DataFrame(dict(c0=C[:, :, 0][sel], c1=C[:, :, 1][sel], o1=o1[sel], po1=po1[sel], p0=P[:, :, 0][sel], p1=P[:, :, 1][sel],
                       c11=np.where(V[:, :, 11] == 1, C[:, :, 11], np.nan)[sel], stay=(G[:, :, 4] == 0)[sel] * 1.0))
df['ratio'] = df.c1 / df.o1.clip(lower=0.5)
for lo, hi in [(0.3, 0.6), (0.6, 0.85)]:
    x0 = df[(df.ratio >= lo) & (df.ratio < hi)]
    print(f'round-1 ratio [{lo},{hi}) n={len(x0)}')
    for a, b in [(0, 0), (1, 10), (11, 20), (21, 30)]:
        x = x0[(x0.p0 >= a) & (x0.p0 <= b)].dropna(subset=['c11'])
        if len(x) < 500: continue
        out = []
        for y in ['c11', 'stay']:
            X = np.column_stack([np.ones(len(x)), x.c0, x.c1, x.o1, x.po1, x.p1])
            bb, *_ = np.linalg.lstsq(X, x[y].values, rcond=None)
            res = x[y].values - X @ bb; se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * res.var())
            out.append(f'{y} per p1 point {bb[5]:+.3f}({se[5]:.3f})')
        print(f'   p0 {a}-{b}: n={len(x)} mean p1 {x.p1.mean():.1f}  ' + '  '.join(out))
