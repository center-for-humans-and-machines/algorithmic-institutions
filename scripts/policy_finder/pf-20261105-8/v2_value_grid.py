"""v2: net value of one punishment point by (own c, round), from v1's sweep.
Y = sum over s>t of sum_i (0.6 c_is - 2 p_is): the group's future net surplus (x4 of the score).
Regress Y on p(t) with state controls (c dummies, c(t-1), others' mean, npun, others' p(t)) and
rule controls (target, log slope, rounds to horizon clipped, log cap) within each (c bin, t bin).
A point at t pays if b > 2 (its own cost 2, in the same x4 units)."""
import numpy as np
z = np.load('/tmp/v1sweep.npz')
sub = np.arange(0, z['C'].shape[0], 2)
C, P, V, R, prm = z['C'][sub].astype(np.float32), z['P'][sub].astype(np.float32), z['V'][sub], z['R'][sub], z['prm']
G = C.shape[0]
net = (0.6 * C - 2 * P) * V           # per player-round
fut = np.cumsum(net.sum(1)[:, ::-1], 1)[:, ::-1]   # fut[:, s] = sum_{u>=s}
fut = np.concatenate([fut[:, 1:], np.zeros((G, 1))], 1)  # sum over u>t
futc = np.cumsum((C * V).sum(1)[:, ::-1], 1)[:, ::-1]; futc = np.concatenate([futc[:, 1:], np.zeros((G, 1))], 1)
npun = np.concatenate([np.zeros((G, 4, 1)), np.cumsum(P > 0, 2)[:, :, :-1]], 2)
cb = [(0, 1), (1, 5), (5, 10), (10, 13), (13, 16), (16, 21)]
tb = [(0, 0), (1, 2), (3, 5), (6, 9), (10, 13), (14, 17), (18, 21)]
print('b = d(future net group surplus)/d(p) ; in brackets d(future sum of contributions)/dp ; point pays if b > 2')
print('c\\t      ' + ' '.join(f'{a}-{b}'.rjust(14) for a, b in tb))
for clo, chi in cb:
    line = f'{clo:2d}-{chi:2d}    '
    for tlo, thi in tb:
        Xs, Ys, Yc = [], [], []
        for t in range(tlo, thi + 1):
            for i in range(4):
                o = [j for j in range(4) if j != i]
                sel = V[:, i, t] & (C[:, i, t] >= clo) & (C[:, i, t] < chi)
                if t > 0: sel &= V[:, i, t - 1]
                g = np.where(sel)[0]
                if len(g) == 0: continue
                cp = C[g, i, t - 1] if t > 0 else np.zeros(len(g))
                Xs.append(np.column_stack([np.ones(len(g)), C[g, i, t], cp, C[g][:, o, t].mean(1), npun[g, i, t],
                                           P[g][:, o, t].mean(1), prm[R[g], 0], np.log(prm[R[g], 1]),
                                           np.clip(prm[R[g], 2] - t, -1, 12), np.log(prm[R[g], 3]), np.full(len(g), t), P[g, i, t]]))
                Ys.append(fut[g, t]); Yc.append(futc[g, t])
        X = np.concatenate(Xs); Y = np.concatenate(Ys); Yc_ = np.concatenate(Yc)
        if X[:, -1].std() < 0.3: line += '            -- '; continue
        b = np.linalg.lstsq(X, Y, rcond=None)[0][-1]; bc = np.linalg.lstsq(X, Yc_, rcond=None)[0][-1]
        line += f'{b:6.1f} [{bc:5.1f}] '
    print(line)
