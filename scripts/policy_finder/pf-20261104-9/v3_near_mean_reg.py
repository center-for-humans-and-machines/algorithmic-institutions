"""Round-0 punishment of near/above-mean members in v3's sweep, with the group's punishment of the
others as a control (herding). OLS of c at round 3 and round 11 (whatever group the player is in)
on own-punishment bins, within ratio buckets, controlling for others' mean punishment and others'
mean contribution in round 0 and the sweep point's rel."""
import numpy as np, pandas as pd, json
z = np.load('/tmp/pf/v3.npz'); C = z['C'].astype(float); V = z['V']; P = z['P'].astype(float); G = z['G']
NP, NE = C.shape[:2]
sw = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v3_sweep/sweep.json'))
pts = sw['points'] if isinstance(sw, dict) else sw
relv = np.zeros(NP)
for q in pts:
    i = int(q['name'].split('_s')[-1]); relv[i] = q['params']['rel']
C = C.reshape(-1, 8, 24); V = V.reshape(-1, 8, 24); P = P.reshape(-1, 8, 24); G = G.reshape(-1, 8, 24)
REL = np.repeat(relv, NE)
t = 0
m = (G[:, :, t] == 0) * V[:, :, t]
k = m.sum(1, keepdims=True)
o = ((C[:, :, t] * m).sum(1, keepdims=True) - C[:, :, t] * m) / np.maximum(k - m, 1)
po = ((P[:, :, t] * m).sum(1, keepdims=True) - P[:, :, t] * m) / np.maximum(k - m, 1)
sel = (m == 1) & (k >= 2)
df = pd.DataFrame(dict(ratio=(C[:, :, 0] / np.maximum(o, 0.5))[sel], c0=C[:, :, 0][sel], o=o.repeat(1, 1)[sel] if False else np.broadcast_to(o, m.shape)[sel],
                       po=np.broadcast_to(po, m.shape)[sel], p=P[:, :, 0][sel], rel=np.broadcast_to(REL[:, None], m.shape)[sel],
                       c3=np.where(V[:, :, 3] == 1, C[:, :, 3], np.nan)[sel], c11=np.where(V[:, :, 11] == 1, C[:, :, 11], np.nan)[sel],
                       c23=np.where(V[:, :, 23] == 1, C[:, :, 23], np.nan)[sel], stay=(G[:, :, 4] == 0)[sel]))
for lo, hi in [(0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.05)]:
    x = df[(df.ratio >= lo) & (df.ratio < hi)].copy()
    x['p1'] = ((x.p > 0) & (x.p <= 5)) * 1.0; x['p2'] = ((x.p > 5) & (x.p <= 12)) * 1.0
    x['p3'] = ((x.p > 12) & (x.p <= 22)) * 1.0; x['p4'] = (x.p > 22) * 1.0
    print(f'ratio [{lo},{hi}) n={len(x)}  share punished {np.mean(x.p>0):.2f}')
    for y in ['c3', 'c11', 'c23', 'stay']:
        xx = x.dropna(subset=[y])
        X = np.column_stack([np.ones(len(xx)), xx.c0, xx.o, xx.po, xx.rel, xx.p1, xx.p2, xx.p3, xx.p4])
        b, *_ = np.linalg.lstsq(X, xx[y].astype(float).values, rcond=None)
        res = xx[y].astype(float).values - X @ b
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var())
        print(f'  {y:5s} po {b[3]:+.3f}({se[3]:.3f}) rel {b[4]:+.2f}  p(0,5] {b[5]:+.2f}({se[5]:.2f}) (5,12] {b[6]:+.2f}({se[6]:.2f}) (12,22] {b[7]:+.2f}({se[7]:.2f}) >22 {b[8]:+.2f}({se[8]:.2f})')
