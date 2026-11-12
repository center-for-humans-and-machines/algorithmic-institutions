"""Follow-up of v3_round0.py: what if every member is punished at round 0? Bins by the smallest dose any member of
my group got (all 4 members punished at least that much) and by the mean dose; w=1 minus w=0, with counts."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/v1.npz'); c, p, m, v = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v']]
P = z['params']; pt = z['point'].astype(int)
k, thr, w = P[pt, 0][:, None], P[pt, 1][:, None], P[pt, 2]
mem = m[:, :, 0] * v[:, :, 0]; n = mem.sum(1, keepdims=True)
mo = ((c[:, :, 0] * mem).sum(1, keepdims=True) - c[:, :, 0] * mem) / np.maximum(n - 1, 1)
dd = np.floor(np.clip(k * np.maximum(thr * mo - c[:, :, 0], 0) * (n - mem > 0), 0, 30)) * mem
mind = np.where(m[:, :, 0] == 1, dd, 99).min(1); meand = dd.sum(1) / np.maximum(m[:, :, 0].sum(1), 1)
pool = (1.6 * c * v - p) * m; opool = (1.6 * c * v - p) * (1 - m)
X = pd.DataFrame(dict(w=w, mind=mind, meand=meand, nval=n[:, 0], c13=(c * m)[:, :, 1:4].sum((1, 2)) / 3,
                      size4=m[:, :, 4].sum(1), leave=(m[:, :, 3] * (1 - m[:, :, 4])).sum(1), join=((1 - m[:, :, 3]) * m[:, :, 4]).sum(1),
                      c4=(c[:, :, 4] * m[:, :, 4]).sum(1), oc4=(c[:, :, 4] * (1 - m[:, :, 4])).sum(1)))
X = X[(X.w <= 1)]
cols = ['c13', 'size4', 'leave', 'join', 'c4', 'oc4']
for name, b in [('mind', [-1, 0, 2, 5, 10, 31]), ('meand', [-1, 0, 2, 5, 10, 20, 31])]:
    X['b'] = pd.cut(X[name], b)
    Y = X[X.nval == 4]
    t = Y.groupby(['b', 'w'], observed=True)[cols].agg(['mean', 'sem']).unstack('w')
    eff = pd.DataFrame({cn: t[(cn, 'mean', 1)] - t[(cn, 'mean', 0)] for cn in cols})
    se = pd.DataFrame({cn + '_se': np.sqrt(t[(cn, 'sem', 1)] ** 2 + t[(cn, 'sem', 0)] ** 2) for cn in ['size4', 'join', 'leave']})
    print(f'by {name} (all 4 members valid)'); print(pd.concat([eff, se], axis=1).round(3).assign(n1=Y[Y.w == 1].groupby('b', observed=True).size(), n0=Y[Y.w == 0].groupby('b', observed=True).size()))
