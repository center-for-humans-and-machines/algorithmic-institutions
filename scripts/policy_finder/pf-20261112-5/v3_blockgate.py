"""v2's `blocks` gate as a natural experiment: at block b (rounds 4b..4b+3), points with blocks==b are closed and
points with blocks>b open, with identical rules before. Prescribed dose at round 4b = D * (#targeted members).
Regress rest-of-game lead (rounds 4b..23) on dose, treatment*dose, interacted with my group size at round 4b."""
import numpy as np, pandas as pd, numpy.linalg as la
z = np.load('/tmp/pf/v2.npz'); c, p, m, v = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v']]
P = z['params']; pt = z['point'].astype(int)
D, lo, hi, w, bl = [P[pt, i] for i in range(5)]
own = (1.6 * c * m * v - p * m).sum(1); oth = (1.6 * c * (1 - m) * v - p * (1 - m)).sum(1)
def ols(X, y, names):
    Xm = np.column_stack([np.ones(len(y))] + X); b, *_ = la.lstsq(Xm, y, rcond=None)
    r = y - Xm @ b; se = np.sqrt(np.diag(r.var() * la.inv(Xm.T @ Xm)))
    return ' '.join(f'{nm}={b[i+1]:+.3f}({se[i+1]:.3f})' for i, nm in enumerate(names))
for b in range(1, 6):
    t = 4 * b
    sel = bl >= b
    tr = (bl > b).astype(float)
    mem = m[:, :, t] * v[:, :, t]; n = m[:, :, t].sum(1)
    tg = (c[:, :, t] >= lo[:, None]) & (c[:, :, t] <= hi[:, None]) & (mem == 1)
    dose = D * tg.sum(1) * (n > 1)
    lead = (own - oth)[:, t:].mean(1)
    nc = n - 4
    X = [tr, dose, tr * dose, nc, tr * nc, tr * dose * nc, w, tr * (w - 1), (c[:, :, t] * m[:, :, t]).sum(1)]
    names = ['tr', 'dose', 'trdose', 'n', 'trn', 'trdose_n', 'w', 'tr_w2', 'sumc']
    print(f'block {b} (N={sel.sum()}) lead:', ols([x[sel] for x in X], lead[sel], names))
    print(f'   members avg after:', ols([x[sel] for x in X], m[:, :, t + 4:].sum(1).mean(1)[sel] if t + 4 < 24 else m[:, :, t].sum(1)[sel], names))
