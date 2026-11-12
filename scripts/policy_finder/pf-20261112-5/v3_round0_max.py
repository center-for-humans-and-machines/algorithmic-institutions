"""v1's round-0 natural experiment (w=1 vs w=0), group level by the largest single dose and the total dose:
treated minus control in joins, leaves and size after the first decision, my contribution in rounds 1-3,
and the other group's contribution in rounds 1-3. Players in my group at round 0 = 4 always (start)."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/v1.npz'); c, p, m, v = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v']]
P = z['params']; pt = z['point'].astype(int)
k, thr, w = P[pt, 0][:, None], P[pt, 1][:, None], P[pt, 2]
mem = m[:, :, 0] * v[:, :, 0]; n = mem.sum(1, keepdims=True)
mo = ((c[:, :, 0] * mem).sum(1, keepdims=True) - c[:, :, 0] * mem) / np.maximum(n - 1, 1)
dd = np.floor(np.clip(k * np.maximum(thr * mo - c[:, :, 0], 0) * (n - mem > 0), 0, 30)) * mem
X = pd.DataFrame(dict(w=w, maxd=dd.max(1), tot=dd.sum(1), c13=(c * m)[:, :, 1:4].sum((1, 2)) / 3, oc13=(c * (1 - m))[:, :, 1:4].sum((1, 2)) / 3,
                      size4=m[:, :, 4].sum(1), leave=(m[:, :, 3] * (1 - m[:, :, 4])).sum(1), join=((1 - m[:, :, 3]) * m[:, :, 4]).sum(1)))
X = X[X.w <= 1]
X['mb'] = pd.cut(X.maxd, [-1, 0, 3, 6, 10, 15, 20, 29, 30])
cols = ['c13', 'oc13', 'size4', 'leave', 'join']
t = X.groupby(['mb', 'w'], observed=True)[cols].agg(['mean', 'sem']).unstack('w')
eff = pd.DataFrame({cn: t[(cn, 'mean', 1)] - t[(cn, 'mean', 0)] for cn in cols})
se = pd.DataFrame({cn + '_se': np.sqrt(t[(cn, 'sem', 1)] ** 2 + t[(cn, 'sem', 0)] ** 2) for cn in ['size4']})
print('by the largest single round-0 dose'); print(pd.concat([eff, se], axis=1).round(3).assign(n=X[X.w == 1].groupby('mb', observed=True).size()))
# member level: own exit at round 3 by own dose, treated - control
rows = []
cc = c[:, :, 0]; left = (m[:, :, 3] == 1) & (m[:, :, 4] == 0)
sel = (mem == 1) & (w[:, None] <= 1)
D = pd.DataFrame(dict(tr=np.broadcast_to(w[:, None] == 1, sel.shape)[sel], d=dd[sel], c0=cc[sel], left=left[sel], c3=c[:, :, 3][sel] - cc[sel]))
D['db'] = pd.cut(D.d, [-1, 0, 5, 10, 20, 29, 30])
t = D.pivot_table(index='db', columns='tr', values=['left', 'c3'], aggfunc='mean', observed=True)
print('member level: own exit and own c change to round 3, treated - control, by own dose')
print(pd.DataFrame({'left': t[('left', True)] - t[('left', False)], 'c3': t[('c3', True)] - t[('c3', False)],
                    'n': D[D.tr].groupby('db', observed=True).size()}).round(3))
