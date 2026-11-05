# v6: shape of the best punishment against the shortfall. Round 0 (and block start 4), v5 sweep:
# within cells (own c, opp, rounded others' mean, group size), by band of the shortfall from v5's
# relative target at w=0.5, c0=10, the lift at t+8 per sqrt point and per point (two fits), and exit.
import numpy as np, sys
def fe(y, x, cell):
    _, inv = np.unique(cell, return_inverse=True); cnt = np.bincount(inv)
    def dm(v): return v - (np.bincount(inv, v) / cnt)[inv]
    X = np.stack([dm(c) for c in x], 1); Y = dm(y)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b; s2 = (r**2).sum() / max(len(Y) - X.shape[1] - len(cnt), 1)
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
d = np.load('/tmp/v5sweep.npz')
C, P, G, V = [d[k].astype(np.float32) for k in 'CPGV']; opp = d['opp']; seat = d['seat']
mine = G == seat[:, None, None]
for t in [0, 4]:
    mv = mine[:, :, t] & (V[:, :, t] > 0)
    s = np.where(mv, C[:, :, t], 0).sum(1, keepdims=True); n = mv.sum(1, keepdims=True)
    om = (s - C[:, :, t]) / np.maximum(n - 1, 1)
    short = 0.5 * 10 + 0.5 * om - C[:, :, t]
    m = mv & (n >= 2)
    st = (G[:, :, t:t + 9] == G[:, :, t:t + 1]).all(2)
    left = (G[:, :, t + 4] != G[:, :, t + 3]).astype(float)
    y = C[:, :, t + 8] - C[:, :, t]
    cell = ((C[:, :, t] * 5 + opp[:, None]) * 25 + np.round(om)) * 9 + n
    if t > 0:
        cell = cell * 7 + np.minimum(np.sqrt(P[:, :, :t].sum(2)) // 2, 6)
    for lo, hi in [(-30, 0), (0, 1.5), (1.5, 3), (3, 5), (5, 7.5), (7.5, 10), (10, 30)]:
        sel = m & (short >= lo) & (short < hi)
        if sel.sum() < 4000: continue
        a, ase = fe(y[sel & st], [np.sqrt(P[:, :, t])[sel & st]], cell[sel & st])
        e, ese = fe(left[sel], [np.sqrt(P[:, :, t])[sel]], cell[sel])
        mp = P[:, :, t][sel].mean()
        print(f't={t} shortfall {lo:5.1f}..{hi:4.1f} n={sel.sum():7d} mean p {mp:5.1f} lift/sqrt p {a[0]:+.3f}({ase[0]:.3f}) exit/sqrt p {e[0]:+.4f}({ese[0]:.4f})')
