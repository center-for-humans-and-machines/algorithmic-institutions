# v4: persistence of a punishment's effect, by the round it is given (v3 sweep): OLS of c[t+h] on our
# punishment at t, controls own c[t], c[t-1], group mean at t, for members staying with us t..t+h.
# Also the spillover: same with the mean punishment of the member's groupmates at t.
import numpy as np
d = np.load('/tmp/v3sweep.npz')
c, p, g, ours = d['c'].astype(np.float32), d['p'].astype(np.float32), d['g'], d['ours']
mine = (g == ours[:, None, None])
pu = p * mine
n = mine.sum(1)
def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + X)
    return np.linalg.lstsq(X, y, rcond=None)[0]
for t in [0, 1, 4, 5, 8]:
    row = []
    for h in [1, 3, 6, 10, 15]:
        if t + h > 23: continue
        sel = np.ones_like(mine[:, :, 0])
        for r in range(t, t + h + 1):
            sel &= mine[:, :, r]
        gm = (c[:, :, t] * mine[:, :, t]).sum(1) / np.maximum(n[:, t], 1)
        gm = np.broadcast_to(gm[:, None], c.shape[:2])
        mates = ((pu[:, :, t].sum(1)[:, None] - pu[:, :, t]) / np.maximum(n[:, t][:, None] - 1, 1))
        prev = c[:, :, t - 1] if t > 0 else c[:, :, t]
        X = [c[:, :, t][sel], prev[sel], gm[sel], pu[:, :, t][sel], mates[sel]]
        b = ols(X, c[:, :, t + h][sel])
        row.append(f'h={h:2d}: own {b[4]:.3f} mates {b[5]:+.3f} (n {sel.sum()//1000}k)')
    print(f't={t}', ' | '.join(row))
