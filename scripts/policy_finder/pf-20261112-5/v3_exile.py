"""Is pushing a low contributor out at a switch decision good for the score? v1's gate at round 3 as a natural
experiment: points with w=4 punish in round 3 (the first decision round), points with w=3 do not, with the same
gating in rounds 0-2. Regress the rest-of-game outcomes (rounds 4-23: own pool, other pool, members) on the dose
v1 prescribes at round 3 to low (c<=lo) and to other members, interacted with treatment."""
import numpy as np, pandas as pd, sys
z = np.load('/tmp/pf/v1.npz'); c, p, m, v = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v']]
P = z['params']; pt = z['point'].astype(int)
k, thr, w = P[pt, 0][:, None], P[pt, 1][:, None], P[pt, 2]
t = 3
mem = m[:, :, t] * v[:, :, t]; n = mem.sum(1, keepdims=True)
mo = ((c[:, :, t] * mem).sum(1, keepdims=True) - c[:, :, t] * mem) / np.maximum(n - 1, 1)
dose = np.floor(np.clip(k * np.maximum(thr * mo - c[:, :, t], 0) * (n - mem > 0), 0, 30)) * mem
sel = (w == 3) | (w == 4)
tr = (w == 4).astype(float)
pool = lambda mask: (1.6 * c * mask * v - p * mask).sum(1)  # [E, 24]
own = pool(m); oth = pool(1 - m)
for LO in [0, 2, 4]:
    low = (c[:, :, t] <= LO) & (mem == 1)
    dl = (dose * low).sum(1); dh = (dose * ~low).sum(1)
    X = pd.DataFrame(dict(tr=tr, dl=dl, dh=dh, trdl=tr * dl, trdh=tr * dh, n=n[:, 0],
                          sc=(c[:, :, :4] * m[:, :, :4]).sum((1, 2)), nlow=low.sum(1)))[sel]
    import numpy.linalg as la
    for name, y in [('own', own[:, 4:].mean(1)), ('oth', oth[:, 4:].mean(1)), ('diff', (own - oth)[:, 4:].mean(1)),
                    ('mem4', m[:, :, 4].sum(1)), ('memavg', m[:, :, 4:].sum(1).mean(1))]:
        Xm = np.column_stack([np.ones(sel.sum()), X.values]); yy = y[sel]
        b, *_ = la.lstsq(Xm, yy, rcond=None)
        r = yy - Xm @ b; s2 = r.var(); cov = s2 * la.inv(Xm.T @ Xm); se = np.sqrt(np.diag(cov))
        cols = ['1'] + list(X.columns)
        print(f'LO={LO} {name}: ' + '  '.join(f'{cn}={b[i]:+.3f}({se[i]:.3f})' for i, cn in enumerate(cols) if cn in ('tr', 'trdl', 'trdh')))
