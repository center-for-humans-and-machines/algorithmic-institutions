"""Does block-1 punishment (v2 blocks gate at round 4) pay differently by my group size at round 4?
Same counterfactual as v3_blockgate2.py, residuals binned by size and by any prescribed dose > 0.
Also block 0 in v1: w>=1 points punish round 0, w=0 points never punish before round 20 -- outcome rounds 0-3 only
and members after the first decision (clean, nothing else differs before round 4)."""
import numpy as np, pandas as pd, numpy.linalg as la, sys
z = np.load('/tmp/pf/v2.npz'); c, p, m, v = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v']]
P = z['params']; pt = z['point'].astype(int)
D, lo, hi, w, bl = [P[pt, i] for i in range(5)]
own = (1.6 * c * m * v - p * m).sum(1); oth = (1.6 * c * (1 - m) * v - p * (1 - m)).sum(1)
for b in [1, 2]:
    t = 4 * b
    sel = bl >= b; tr = bl > b
    mem = m[:, :, t] * v[:, :, t]; n = m[:, :, t].sum(1)
    tg = ((c[:, :, t] >= lo[:, None]) & (c[:, :, t] <= hi[:, None]) & (mem == 1)).sum(1) * (n > 1)
    dose = D * tg
    feats = [n, n ** 2, (c[:, :, t] * m[:, :, t]).sum(1), (c[:, :, t] * (1 - m[:, :, t])).sum(1),
             (own - oth)[:, t - 4:t].mean(1), own[:, t - 4:t].mean(1), D, np.log(D), tg, dose, w, tg * w, dose * w, lo, hi]
    Xf = np.column_stack([np.ones(len(n))] + feats)
    for name, y in [('lead', (own - oth)[:, t:].sum(1)), ('own', own[:, t:].sum(1)), ('oth', oth[:, t:].sum(1))]:
        ctl = sel & ~tr
        bb, *_ = la.lstsq(Xf[ctl], y[ctl], rcond=None)
        r = y - Xf @ bb
        T = pd.DataFrame(dict(r=r[sel], tr=tr[sel], nb=pd.cut(n[sel], [-1, 3, 5, 8]), db=pd.cut(dose[sel], [-1, 0, 15, 100])))
        a = T[T.tr].groupby(['nb', 'db'], observed=True).r.agg(['mean', 'sem', 'size'])
        b_ = T[~T.tr].groupby(['nb', 'db'], observed=True).r.agg(['mean', 'sem'])
        out = pd.DataFrame({'eff': a['mean'] - b_['mean'], 'se': np.sqrt(a['sem'] ** 2 + b_['sem'] ** 2), 'n': a['size']})
        print(f'block {b} {name}:'); print(out.round(1).unstack('db'))
