# v5: robustness of exploration 4: round-0 sqrt-p slope estimated separately by gap band (others' mean - own c)
import numpy as np
def fe(y, x, cell):
    _, inv = np.unique(cell, return_inverse=True); cnt = np.bincount(inv)
    def dm(v): return v - (np.bincount(inv, v) / cnt)[inv]
    X = np.stack([dm(c) for c in x], 1); Y = dm(y)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b; s2 = (r**2).sum() / (len(Y) - X.shape[1] - len(cnt))
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
Y8, ST, SQ, GAP, CL, CELL, SW = [], [], [], [], [], [], []
for v in ['v1', 'v2', 'v3', 'v4']:
    d = np.load(f'/tmp/{v}sweep.npz')
    C, P, G, V = [d[k].astype(np.float32) for k in 'CPGV']; opp = d['opp']; seat = d['seat']
    mine = G == seat[:, None, None]; t = 0
    mv = mine[:, :, t] & (V[:, :, t] > 0)
    s = np.where(mv, C[:, :, t], 0).sum(1, keepdims=True); n = mv.sum(1, keepdims=True)
    om = (s - C[:, :, t]) / np.maximum(n - 1, 1)
    m = mv & (C[:, :, t] <= 12) & (n >= 2)
    st8 = (G[:, :, :9] == G[:, :, :1]).all(2)
    Y8.append((C[:, :, 8] - C[:, :, 0])[m]); ST.append(st8[m]); SQ.append(np.sqrt(P[:, :, 0])[m]); GAP.append((om - C[:, :, 0])[m])
    CL.append(C[:, :, 0][m]); SW.append((G[:, :, 4] != G[:, :, 3])[m].astype(float))
    CELL.append(((C[:, :, 0] * 5 + opp[:, None]) * 25 + np.round(om))[m] * 9 + np.broadcast_to(n, m.shape)[m])
    del C, P, G, V
Y8, ST, SQ, GAP, CL, CELL, SW = map(np.concatenate, (Y8, ST, SQ, GAP, CL, CELL, SW))
for lo, hi in [(-20, 0), (0, 2), (2, 4), (4, 6), (6, 9), (9, 21)]:
    for clo, chi in [(0, 4), (5, 8), (9, 12)]:
        s = (GAP >= lo) & (GAP < hi) & (CL >= clo) & (CL <= chi)
        if s.sum() < 5000: continue
        b, se = fe(Y8[s & ST], [SQ[s & ST]], CELL[s & ST]); b2, se2 = fe(SW[s], [SQ[s]], CELL[s])
        print('gap %3d..%3d own c %2d-%2d n=%7d  dc(t+8) per sqrt p %+.3f(%.3f)  exit %+.4f(%.4f)' % (lo, hi, clo, chi, s.sum(), b[0], se[0], b2[0], se2[0]))
