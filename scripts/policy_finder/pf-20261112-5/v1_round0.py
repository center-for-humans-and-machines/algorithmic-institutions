"""Clean causal read from v1: round 0 is played before any manager acts, so its state has the same law at
every point. Points with w=0 never punish in rounds 0-3; points with w=1 punish only in round 0 of the block.
For each member of the rule's group, compute the dose v1's formula prescribes in round 0
(floor(k*relu(thr*mean_others - c))); for w=1 it was dealt, for w=0 not. Compare, at equal prescribed dose,
the member's contributions in rounds 1-3, its exit at the round-3 decision, and the rest of the group's."""
import numpy as np, pandas as pd
z = np.load('/tmp/v1_compact.npz'); c, p, m, v = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v']]
P = z['params']; pt = z['point'].astype(int)
k, thr, w = P[pt, 0][:, None], P[pt, 1][:, None], P[pt, 2][:, None]
mem = m[:, :, 0] * v[:, :, 0]
n = mem.sum(1, keepdims=True)
mo = ((c[:, :, 0] * mem).sum(1, keepdims=True) - c[:, :, 0] * mem) / np.maximum(n - 1, 1)
dose = np.floor(np.clip(k * np.maximum(thr * mo - c[:, :, 0], 0) * (n - mem > 0), 0, 30)) * mem
mm = dose[(w[:, 0] == 1)] != p[w[:, 0] == 1][:, :, 0]; print("dose mismatch share", mm.mean(), np.abs(dose[(w[:, 0] == 1)] - p[w[:, 0] == 1][:, :, 0]).max())
gdose = dose.sum(1, keepdims=True)
sel = (mem == 1) & np.isin(w, [0, 1])
treated = np.broadcast_to(w == 1, sel.shape)
X = pd.DataFrame(dict(tr=treated[sel], dose=dose[sel], gdose=np.broadcast_to(gdose, sel.shape)[sel] - dose[sel],
                      c0=c[:, :, 0][sel], c1=c[:, :, 1][sel], c2=c[:, :, 2][sel], c3=c[:, :, 3][sel],
                      c7=np.where(m[:, :, 7] == 1, c[:, :, 7], np.nan)[sel],
                      left=((m[:, :, 3] == 1) & (m[:, :, 4] == 0))[sel]))
X['db'] = pd.cut(X.dose, [-1, 0, 2, 5, 10, 20, 30])
X['cb'] = pd.cut(X.c0, [-1, 0, 4, 9, 14, 20])
for col in ['c1', 'c3', 'left']:
    t = X.pivot_table(index=['cb', 'db'], columns='tr', values=col, aggfunc='mean', observed=True)
    t['diff'] = t[True] - t[False]
    t['n_tr'] = X[X.tr].groupby(['cb', 'db'], observed=True).size()
    print(col); print(t.round(3))
# spillover on unpunished members (own prescribed dose 0) by the group's prescribed dose to others
U = X[X.dose == 0].copy(); U['gb'] = pd.cut(U.gdose, [-1, 0, 5, 15, 40, 300])
for col in ['c1', 'c3', 'left']:
    t = U.pivot_table(index='gb', columns='tr', values=col, aggfunc='mean', observed=True); t['diff'] = t[True] - t[False]
    print('unpunished members,', col, 'by others prescribed dose'); print(t.round(3))
