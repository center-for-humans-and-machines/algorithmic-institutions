"""Gate natural experiment (see v1_gate.py) for members prescribed no punishment: their next-round and +4
contribution change and exit, when the group-mates' prescribed total dose this round is in a bin."""
import numpy as np, pandas as pd
z = np.load('/tmp/v1_compact.npz'); c, p, m, v = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v']]
P = z['params']; pt = z['point'].astype(int)
k, thr, w = P[pt, 0][:, None], P[pt, 1][:, None], P[pt, 2][:, None]
rows = []
for t in range(1, 20):
    pos = t % 4; dec = t - pos + 3
    mem = m[:, :, t] * v[:, :, t]; n = mem.sum(1, keepdims=True)
    mo = ((c[:, :, t] * mem).sum(1, keepdims=True) - c[:, :, t] * mem) / np.maximum(n - 1, 1)
    dose = np.floor(np.clip(k * np.maximum(thr * mo - c[:, :, t], 0) * (n - mem > 0), 0, 30)) * mem
    gd = dose.sum(1, keepdims=True) - dose
    sel = (mem == 1) & ((w == pos) | (w == pos + 1)) & (dose == 0)
    tr = np.broadcast_to(w == pos + 1, sel.shape)
    c4 = np.where(m[:, :, t + 4] == 1, c[:, :, t + 4], np.nan)
    rows.append(pd.DataFrame(dict(tr=tr[sel], gd=np.broadcast_to(gd, sel.shape)[sel], c0=c[:, :, t][sel],
                                  c1=c[:, :, t + 1][sel] - c[:, :, t][sel], c4=(c4 - c[:, :, t])[sel],
                                  left=((m[:, :, dec] == 1) & (m[:, :, dec + 1] == 0))[sel])))
X = pd.concat(rows); X['gb'] = pd.cut(X.gd, [-1, 0, 5, 10, 20, 40, 300])
t = X.pivot_table(index='gb', columns='tr', values=['c1', 'c4', 'left'], aggfunc='mean', observed=True)
print(pd.DataFrame({col: t[(col, True)] - t[(col, False)] for col in ['c1', 'c4', 'left']}).round(3).assign(n=X[X.tr].groupby('gb', observed=True).size()))
