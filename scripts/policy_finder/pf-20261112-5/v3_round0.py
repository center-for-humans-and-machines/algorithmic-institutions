"""v1's round-0 natural experiment (w=1 punishes round 0 and no other round of block 0; w=0 never before 20), at the
group level: by the total dose v1 prescribes my group at round 0, treated minus control in my group's contribution
in rounds 1-3, its size after the first decision, joins and leaves at that decision, and the lead in rounds 0-3 and
4-7 (w=1 also punishes round 4, so 4-7 mixes it in). Uses /tmp/pf/v1.npz from compact.py."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/v1.npz'); c, p, m, v = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v']]
P = z['params']; pt = z['point'].astype(int)
k, thr, w = P[pt, 0][:, None], P[pt, 1][:, None], P[pt, 2]
mem = m[:, :, 0] * v[:, :, 0]; n = mem.sum(1, keepdims=True)
mo = ((c[:, :, 0] * mem).sum(1, keepdims=True) - c[:, :, 0] * mem) / np.maximum(n - 1, 1)
dd = np.floor(np.clip(k * np.maximum(thr * mo - c[:, :, 0], 0) * (n - mem > 0), 0, 30)) * mem
dose = dd.sum(1); ntg = (dd > 0).sum(1)
pool = (1.6 * c * v - p) * m; opool = (1.6 * c * v - p) * (1 - m)
X = pd.DataFrame(dict(w=w, dose=dose, ntg=ntg, c13=(c * m)[:, :, 1:4].sum((1, 2)) / 3, oc13=(c * (1 - m))[:, :, 1:4].sum((1, 2)) / 3,
                      size4=m[:, :, 4].sum(1), leave=(m[:, :, 3] * (1 - m[:, :, 4])).sum(1), join=((1 - m[:, :, 3]) * m[:, :, 4]).sum(1),
                      lead03=(pool - opool)[:, :, 0:4].sum((1, 2)) / 4, lead47=(pool - opool)[:, :, 4:8].sum((1, 2)) / 4,
                      maxd=dd.max(1)))
X = X[X.w <= 1]
X['db'] = pd.cut(X.dose, [-1, 0, 5, 10, 20, 40, 300])
cols = ['c13', 'oc13', 'size4', 'leave', 'join', 'lead03', 'lead47']
t = X.groupby(['db', 'w'], observed=True)[cols].mean().unstack('w')
eff = pd.DataFrame({cn: t[(cn, 1)] - t[(cn, 0)] for cn in cols})
se = X.groupby(['db', 'w'], observed=True)[cols].sem().unstack('w')
ses = pd.DataFrame({cn: np.sqrt(se[(cn, 1)] ** 2 + se[(cn, 0)] ** 2) for cn in cols})
print('effect (w=1 minus w=0) by total round-0 dose'); print(eff.round(3)); print('se'); print(ses.round(3))
print(X.groupby(['db', 'w'], observed=True).size().unstack())
X['tb'] = X.ntg.clip(0, 4)
t = X[X.dose > 20].groupby(['tb', 'w'], observed=True)[cols].mean().unstack('w')
print('dose>20, by number of targets'); print(pd.DataFrame({cn: t[(cn, 1)] - t[(cn, 0)] for cn in cols}).round(3))
