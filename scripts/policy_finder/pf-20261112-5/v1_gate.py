"""Natural experiment in v1's gate: at round t<20 with block position pos=t%4, points with w=pos+1 punish
(this round is their last open one), points with w=pos do not (closed from this round on), with otherwise the
same history of gating. At equal prescribed dose and own contribution, compare: next-round contribution, the
member's exit at the block's decision, its contribution 4 rounds later (if still a member), by pos."""
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
    sel = (mem == 1) & ((w == pos) | (w == pos + 1)) & (dose > 0)
    tr = np.broadcast_to(w == pos + 1, sel.shape)
    c4 = np.where(m[:, :, t + 4] == 1, c[:, :, t + 4], np.nan)
    rows.append(pd.DataFrame(dict(t=t, pos=pos, tr=tr[sel], dose=dose[sel], c0=c[:, :, t][sel], c1=c[:, :, t + 1][sel] - c[:, :, t][sel],
                                  c4=(c4 - c[:, :, t])[sel], left=((m[:, :, dec] == 1) & (m[:, :, dec + 1] == 0))[sel])))
X = pd.concat(rows)
X['db'] = pd.cut(X.dose, [0, 3, 8, 15, 30]); X['cb'] = pd.cut(X.c0, [-1, 0, 3, 7, 11, 20])
for col in ['c1', 'c4', 'left']:
    t = X.pivot_table(index=['cb', 'db'], columns='tr', values=col, aggfunc='mean', observed=True)
    print(col, '(treated - control) by own c (rows) and dose; all positions'); print((t[True] - t[False]).unstack('db').round(2))
t = X.pivot_table(index=['pos', 'db'], columns='tr', values='left', aggfunc='mean', observed=True)
print('exit difference by block position and dose'); print((t[True] - t[False]).unstack('db').round(3))
print('n treated by cb, db'); print(X[X.tr].groupby(['cb', 'db'], observed=True).size().unstack())
# does the answer depend on the others' mean at equal own contribution? (absolute vs relative target)
rows = []
for t in range(1, 20):
    pos = t % 4
    mem = m[:, :, t] * v[:, :, t]; n = mem.sum(1, keepdims=True)
    mo = ((c[:, :, t] * mem).sum(1, keepdims=True) - c[:, :, t] * mem) / np.maximum(n - 1, 1)
    dose = np.floor(np.clip(k * np.maximum(thr * mo - c[:, :, t], 0) * (n - mem > 0), 0, 30)) * mem
    sel = (mem == 1) & ((w == pos) | (w == pos + 1)) & (dose > 2) & (dose <= 10)
    tr = np.broadcast_to(w == pos + 1, sel.shape)
    c4 = np.where(m[:, :, t + 4] == 1, c[:, :, t + 4], np.nan)
    rows.append(pd.DataFrame(dict(tr=tr[sel], c0=c[:, :, t][sel], mo=np.broadcast_to(mo, sel.shape)[sel], c4=(c4 - c[:, :, t])[sel],
                                  rem=23 - t + 0 * c[:, :, t][sel])))
Y = pd.concat(rows)
Y['cb'] = pd.cut(Y.c0, [-1, 0, 3, 7, 11, 20]); Y['mb'] = pd.cut(Y.mo, [0, 6, 9, 12, 20])
t = Y.pivot_table(index='cb', columns=['mb', 'tr'], values='c4', aggfunc='mean', observed=True)
print('dose 3-10: c4 effect by own c (rows) and others mean (cols)')
print(pd.DataFrame({mb: t[(mb, True)] - t[(mb, False)] for mb in Y.mb.cat.categories if (mb, True) in t}).round(2))
