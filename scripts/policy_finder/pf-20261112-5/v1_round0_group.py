"""Group-level version of v1_round0.py: per game, the total dose v1's formula prescribes to my group in round 0;
w=1 games dealt it, w=0 games did not. Compare my group's contribution sum in rounds 1-3, its size after the
round-3 decision, the other group's size, and the lead (pool difference) in rounds 4-7 and 8-23 (w=1 games also
punish later, so the later columns mix in later punishment)."""
import numpy as np, pandas as pd
z = np.load('/tmp/v1_compact.npz'); c, p, m, v = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v']]
P = z['params']; pt = z['point'].astype(int)
k, thr, w = P[pt, 0][:, None], P[pt, 1][:, None], P[pt, 2]
mem = m[:, :, 0] * v[:, :, 0]; n = mem.sum(1, keepdims=True)
mo = ((c[:, :, 0] * mem).sum(1, keepdims=True) - c[:, :, 0] * mem) / np.maximum(n - 1, 1)
dose = (np.floor(np.clip(k * np.maximum(thr * mo - c[:, :, 0], 0) * (n - mem > 0), 0, 30)) * mem).sum(1)
pool = (1.6 * c - p) * m; opool = (1.6 * c - p) * (1 - m)
X = pd.DataFrame(dict(w=w, dose=dose, c13=(c * m)[:, :, 1:4].sum((1, 2)), size4=m[:, :, 4].sum(1),
                      lead47=(pool - opool)[:, :, 4:8].sum((1, 2)) / 4, lead823=(pool - opool)[:, :, 8:].sum((1, 2)) / 16,
                      lead=(pool - opool).sum((1, 2)) / 24))
X = X[X.w <= 1]
X['db'] = pd.cut(X.dose, [-1, 0, 5, 10, 20, 40, 300])
t = X.groupby(['db', 'w'], observed=True)[['c13', 'size4', 'lead47', 'lead823', 'lead']].mean().unstack('w')
print(t.round(2))
print(X.groupby(['db', 'w'], observed=True).size().unstack())
