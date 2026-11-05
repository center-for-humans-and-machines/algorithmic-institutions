# v2: how the artificial humans' contributions answer punishment (from v1's simulated games)
import numpy as np
d = np.load('/tmp/v1sweep.npz'); C, P, G, V = [d[k].astype(np.float32) for k in 'CPGV']
rng = np.random.default_rng(0); idx = rng.choice(len(C), 40000, replace=False)
C, P, G, V = C[idx], P[idx], G[idx], V[idx]
# one-step response by own contribution band and punishment band, players staying in group
same = G[:, :, 1:] == G[:, :, :-1]
c0, c1, p0 = C[:, :, :-1], C[:, :, 1:], P[:, :, :-1]
ok = same & (V[:, :, :-1] > 0) & (V[:, :, 1:] > 0)
rd = np.broadcast_to(np.arange(23), c0.shape)
print('mean dc next round by c band (rows) x p band (cols)')
pb = [(0,0),(1,2),(3,5),(6,9),(10,14),(15,30)]
for lo, hi in [(0,4),(5,9),(10,14),(15,19),(20,20)]:
    row = []
    for a, b in pb:
        m = ok & (c0 >= lo) & (c0 <= hi) & (p0 >= a) & (p0 <= b)
        row.append(f'{(c1-c0)[m].mean():+.2f}({m.sum()})' if m.sum() > 50 else '   .   ')
    print(f'c {lo:2d}-{hi:2d}', ' '.join(row))
# persistence: c(t+k) - c(t) for round-0 punishment
print('\nround-0 punishment band -> mean contribution in later rounds (stayers in same group all along until r)')
for a, b in pb:
    m0 = (P[:, :, 0] >= a) & (P[:, :, 0] <= b) & (C[:, :, 0] <= 9)
    vals = [C[:, :, r][m0 & (G[:, :, r] == G[:, :, 0])].mean() for r in [0, 1, 2, 3, 5, 8, 12, 16, 23]]
    print(f'p0 {a:2d}-{b:2d} n={m0.sum():6d}', np.round(vals, 2))
# regression: c(t+k) on c(t), mean others c(t), p(t), for k=1,3,6
for k in [1, 3, 6]:
    X, Y = [], []
    for t in range(0, 24 - k):
        oth = (C[:, :, t].sum(1, keepdims=True) - C[:, :, t])  # crude: all others both groups
        m = (G[:, :, t+k] == G[:, :, t]) & (V[:, :, t] > 0) & (V[:, :, t+k] > 0)
        grp = np.stack([((G[:, :, t] == g) * C[:, :, t]).sum(1) / np.maximum((G[:, :, t] == g).sum(1), 1) for g in [0, 1]], 1)
        gm = np.take_along_axis(grp, G[:, :, t].astype(int), 1)
        X.append(np.stack([np.ones(m.sum()), C[:, :, t][m], gm[m], P[:, :, t][m], P[:, :, t][m] * (C[:, :, t][m] >= 12), np.full(m.sum(), t)], 1)); Y.append(C[:, :, t+k][m])
    X, Y = np.concatenate(X), np.concatenate(Y)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    print(f'k={k}: const {b[0]:.2f} own {b[1]:.3f} groupmean {b[2]:.3f} p {b[3]:.3f} p*high {b[4]:.3f} t {b[5]:.3f}')
