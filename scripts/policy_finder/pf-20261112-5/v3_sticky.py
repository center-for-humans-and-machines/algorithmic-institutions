"""Are there sticky members? In the reference sims and v1's games, switch rate at decision d by the member's
history: tenure in its current group (blocks since last switch, or since start) and whether it ever switched.
Then v1's gate natural experiment (w=pos+1 vs w=pos) for the exit effect of a dose, by that history."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/v1.npz'); c, p, m, v, g = [z[k].astype(np.float64) for k in ['c', 'p', 'mine', 'v', 'g']]
P = z['params']; pt = z['point'].astype(int)
k, thr, w = P[pt, 0][:, None], P[pt, 1][:, None], P[pt, 2][:, None]
rows = []
for dec in [7, 11, 15, 19]:
    # number of switches before (decisions up to dec-4)
    nsw = sum((g[:, :, d] != g[:, :, d + 1]) for d in range(3, dec, 4))
    joined_last = (g[:, :, dec - 4] != g[:, :, dec - 3])
    for t in range(dec - 3, dec + 1):
        pos = t % 4
        mem = m[:, :, t] * v[:, :, t]; n = mem.sum(1, keepdims=True)
        mo = ((c[:, :, t] * mem).sum(1, keepdims=True) - c[:, :, t] * mem) / np.maximum(n - 1, 1)
        dose = np.floor(np.clip(k * np.maximum(thr * mo - c[:, :, t], 0) * (n - mem > 0), 0, 30)) * mem
        sel = (m[:, :, dec] == 1) & (mem == 1) & ((w == pos) | (w == pos + 1))
        tr = np.broadcast_to(w == pos + 1, sel.shape)
        rows.append(pd.DataFrame(dict(dec=dec, pos=pos, tr=tr[sel], dose=dose[sel], nsw=nsw[sel], jl=joined_last[sel],
                                      left=(g[:, :, dec] != g[:, :, dec + 1])[sel])))
X = pd.concat(rows)
X['hist'] = np.where(X.jl, 'joined_last', np.where(X.nsw == 0, 'never_switched', 'switched_earlier'))
print('base exit rate (control, dose 0 or untreated) by history and decision:')
print(X[~X.tr].groupby(['hist', 'dec']).left.mean().unstack().round(3))
X['db'] = pd.cut(X.dose, [-1, 0, 3, 8, 15, 30])
t = X.pivot_table(index=['hist', 'db'], columns='tr', values='left', aggfunc='mean', observed=True)
print('exit: treated - control, by history and dose (all positions)'); print((t[True] - t[False]).unstack('db').round(3))
