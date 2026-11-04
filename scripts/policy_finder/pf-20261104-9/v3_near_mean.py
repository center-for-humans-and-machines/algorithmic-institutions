"""Do near- or above-mean members gain from a punishment in the sim? v3 sweep, my group (0),
rounds 0, 4, 8 (fresh rounds). Members bucketed by c / others' mean; outcomes: change in c to
the round before the next decision (t+3), and whether they are still in my group after it (t+4).
Within a bucket punishment depends mainly on the sweep's rel/slope/fresh, so the comparison is
close to a design-driven one."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/v3.npz'); C = z['C'].astype(float); V = z['V']; P = z['P'].astype(float); G = z['G']
NP, NE = C.shape[:2]
C = C.reshape(-1, 8, 24); V = V.reshape(-1, 8, 24); P = P.reshape(-1, 8, 24); G = G.reshape(-1, 8, 24)
m = (G == 0) * V
s = (C * m).sum(1, keepdims=True); k = m.sum(1, keepdims=True)
o = (s - C * m) / np.maximum(k - m, 1)
rows = []
for t in (0, 4, 8):
    sel = (G[:, :, t] == 0) & (V[:, :, t] == 1) & (k[:, 0, t][:, None] >= 2)
    stay = (G[:, :, t + 4] == 0)
    ok3 = (G[:, :, t + 3] == 0) & (V[:, :, t + 3] == 1)
    rows.append(pd.DataFrame(dict(t=t, ratio=(C[:, :, t] / np.maximum(o[:, 0, t][:, None], 0.5))[sel],
                                  p=P[:, :, t][sel], stay=stay[sel], dc=np.where(ok3, C[:, :, t + 3] - C[:, :, t], np.nan)[sel])))
r = pd.concat(rows)
r['rb'] = pd.cut(r.ratio, [0, 0.5, 0.8, 0.95, 1.05, 1.2, 1.5, 100], right=False)
r['pb'] = pd.cut(r.p, [-1, 0, 2, 5, 10, 20, 30])
pd.set_option('display.width', 250)
for t in (0, 4, 8):
    x = r[r.t == t]
    print('round', t)
    print(x.groupby(['rb', 'pb'], observed=True).agg(dc=('dc', 'mean'), stay=('stay', 'mean'), n=('p', 'size')).round(2).unstack('pb').to_string())
