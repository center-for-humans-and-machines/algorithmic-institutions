# AH response in v1's sweep: next-round contribution vs own punishment, by own contribution bin;
# and persistence (c at t+1..t+4) for players staying in our group.
import numpy as np
d = np.load('/tmp/v1sweep.npz')
c, p, v, g, ours = d['c'].astype(np.float32), d['p'].astype(np.float32), d['v'], d['g'], d['ours']
ing = (g == ours[:, None, None])
cnt = ing.sum(1, keepdims=True)
gm = (c * ing).sum(1, keepdims=True) / np.maximum(cnt, 1)
# within block rounds that are followed by a non-switch round: t in {0,1,2, 4,5,6, ...}
rows = []
for t in range(23):
    if t % 4 == 3: continue
    m = ing[:, :, t] & ing[:, :, t + 1] & v[:, :, t] & v[:, :, t + 1]
    rows.append(np.stack([c[:, :, t][m], p[:, :, t][m], c[:, :, t + 1][m], gm[:, 0, t][np.nonzero(m)[0]],
                          np.full(m.sum(), t)], 1))
R = np.concatenate(rows)
ct, pt, cn, gmt, tt = R.T
print('n', len(R))
pb = [(-1, 0), (0, 2), (2, 5), (5, 10), (10, 20), (20, 31)]
print('next-round change by own c bin (rows) and punishment bin (cols)')
for lo, hi in [(0, 0), (1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]:
    mm = (ct >= lo) & (ct <= hi)
    out = []
    for a, b in pb:
        k = mm & (pt > a) & (pt <= b)
        out.append(f'{(cn[k]-ct[k]).mean():6.2f}({k.sum():6d})' if k.sum() > 50 else '     -        ')
    print(f'c {lo:2d}-{hi:2d}', ' '.join(out))
# OLS with controls
X = np.stack([np.ones_like(ct), ct, pt, gmt, tt], 1)
for lo, hi in [(0, 4), (5, 9), (10, 14), (15, 20)]:
    mm = (ct >= lo) & (ct <= hi)
    b = np.linalg.lstsq(X[mm], cn[mm], rcond=None)[0]
    print(f'OLS c{lo}-{hi}: const {b[0]:.2f} c {b[1]:.3f} p {b[2]:.3f} gmean {b[3]:.3f} t {b[4]:.3f}')
# persistence: players in our group rounds t..t+h; effect of p at t on c at t+h, block start rounds
print('persistence: coef on p[t] for c[t+h], controls c[t], gmean[t]; low contributors c<=8, t in {0,4,8,12}')
for h in range(1, 9):
    rr = []
    for t in [0, 4, 8, 12]:
        if t + h > 23: continue
        m = ing[:, :, t:t + h + 1].all(2) & v[:, :, t] & v[:, :, t + h] & (c[:, :, t] <= 8)
        rr.append(np.stack([np.ones(m.sum()), c[:, :, t][m], p[:, :, t][m], gm[:, 0, t][np.nonzero(m)[0]], c[:, :, t + h][m]], 1))
    rr = np.concatenate(rr)
    b = np.linalg.lstsq(rr[:, :4], rr[:, 4], rcond=None)[0]
    print(f' h={h}: p coef {b[2]:.3f}  c coef {b[1]:.3f}  n={len(rr)}')
