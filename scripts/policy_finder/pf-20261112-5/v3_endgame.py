"""Block 5 (rounds 20-23, nobody can switch any more) in v2: points with blocks=6 punish at round 20 (and 21 if w=2),
points with blocks=5 do not. Nonparametric: by my group size n and the round-20 prescribed number of targets,
treated minus control in own pool, other pool, own contributions, rounds 20-23 and 21-23."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/v2.npz'); c, p, m, v = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v']]
P = z['params']; pt = z['point'].astype(int)
D, lo, hi, w, bl = [P[pt, i] for i in range(5)]
t = 20
sel = bl >= 5; tr = bl > 5
mem = m[:, :, t] * v[:, :, t]; n = m[:, :, t].sum(1)
tg = (c[:, :, t] >= lo[:, None]) & (c[:, :, t] <= hi[:, None]) & (mem == 1)
X = pd.DataFrame(dict(tr=tr, n=n, ntg=tg.sum(1) * (n > 1), D=D, w=w,
    own=(1.6 * c * m * v - p * m).sum(1)[:, 20:].mean(1), oth=(1.6 * c * (1 - m) * v - p * (1 - m)).sum(1)[:, 20:].mean(1),
    ownc=(c * m).sum(1)[:, 21:].mean(1), othc=(c * (1 - m)).sum(1)[:, 21:].mean(1), pun=(p * m).sum(1)[:, 20:].sum(1)))[sel]
X['Db'] = pd.cut(X.D, [0, 2.5, 6, 12])
for col in ['own', 'oth', 'ownc', 'othc', 'pun']:
    tt = X.pivot_table(index=['n'], columns=['tr'], values=col, aggfunc='mean', observed=True)
    print(col, 'by n (treated-control):', (tt[True] - tt[False]).round(2).to_dict())
X['lead'] = X.own - X.oth
tt = X[X.ntg > 0].pivot_table(index=['Db'], columns=['tr'], values=['lead', 'ownc', 'pun'], aggfunc='mean', observed=True)
print(tt.round(2))
print(X.groupby(['n', 'tr']).size().unstack())
