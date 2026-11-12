"""v2's `blocks` gate, counterfactual version: fit the rest-of-game outcome on the control arm (blocks==b, closed
from block b on) from the state at round 4b (my size, both groups' contributions at 4b, pre-block lead, D, the
prescribed targets, w); predict it for the treated arm (blocks>b) and average treated minus control residuals by prescribed dose
(D * targets at round 4b). Outcomes: lead over rounds 4b..23 summed (pool points per game), own and other pool."""
import numpy as np, pandas as pd, numpy.linalg as la
z = np.load('/tmp/pf/v2.npz'); c, p, m, v = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v']]
P = z['params']; pt = z['point'].astype(int)
D, lo, hi, w, bl = [P[pt, i] for i in range(5)]
own = (1.6 * c * m * v - p * m).sum(1); oth = (1.6 * c * (1 - m) * v - p * (1 - m)).sum(1)
for b in range(1, 6):
    t = 4 * b
    sel = bl >= b; tr = bl > b
    mem = m[:, :, t] * v[:, :, t]; n = m[:, :, t].sum(1)
    tg = ((c[:, :, t] >= lo[:, None]) & (c[:, :, t] <= hi[:, None]) & (mem == 1)).sum(1) * (n > 1)
    dose = D * tg
    feats = [n, n ** 2, (c[:, :, t] * m[:, :, t]).sum(1), (c[:, :, t] * (1 - m[:, :, t])).sum(1),
             (own - oth)[:, t - 4:t].mean(1), own[:, t - 4:t].mean(1), D, np.log(D), tg, dose, w, tg * w, dose * w, lo, hi]
    Xf = np.column_stack([np.ones(len(n))] + feats)
    res = {}
    for name, y in [('lead', (own - oth)[:, t:].sum(1)), ('own', own[:, t:].sum(1)), ('oth', oth[:, t:].sum(1)),
                    ('mem_end', m[:, :, 23].sum(1))]:
        ctl = sel & ~tr
        bb, *_ = la.lstsq(Xf[ctl], y[ctl], rcond=None)
        r = y - Xf @ bb
        T = pd.DataFrame(dict(r=r[sel], tr=tr[sel], db=pd.cut(dose[sel], [-1, 0, 5, 10, 20, 40, 100])))
        a = T[T.tr].groupby('db', observed=True).r.agg(['mean', 'sem', 'size'])
        b_ = T[~T.tr].groupby('db', observed=True).r.agg(['mean', 'sem', 'size'])
        res[name] = pd.DataFrame({'mean': a['mean'] - b_['mean'], 'sem': np.sqrt(a['sem'] ** 2 + b_['sem'] ** 2), 'size': a['size']})
    print(f'--- block {b} (rounds {t}-23): treated residual vs control-fitted, by prescribed dose at round {t}')
    print(pd.concat({k: v_[['mean', 'sem']] for k, v_ in res.items()}, axis=1).round(2).assign(n=res['lead']['size']))
