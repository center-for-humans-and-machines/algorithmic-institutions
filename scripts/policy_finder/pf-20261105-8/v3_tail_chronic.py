"""v3: late (t 9-16) punishment of low contributors (c<6) split by the player's earlier punishment
(cumulative points before t): does the tail still pay on chronic free riders? Same design as v3_late_dose.py."""
import numpy as np
rows = []
for fam, f in enumerate(('/tmp/v1sweep.npz', '/tmp/v2sweep.npz')):
    z = np.load(f); C, P, V, R, prm = z['C'].astype(np.float32), z['P'].astype(np.float32), z['V'], z['R'], z['prm']
    if fam == 0: pr = np.column_stack([prm[:, 0], np.log(prm[:, 1]), np.minimum(prm[:, 2], 24), np.log(prm[:, 3])])
    else: pr = np.column_stack([prm[:, 0], prm[:, 1], np.log(prm[:, 2]), np.minimum(prm[:, 3], 24)])
    for t in range(9, 17):
        for i in range(4):
            o = [j for j in range(4) if j != i]
            g = np.where(V[:, :, t].all(1) & V[:, i, t + 1:].all(1) & (C[:, i, t] < 6))[0]
            own = C[g, i, t + 1:].mean(1) - C[g, i, t]; oth = C[g][:, o, t + 1:].mean((1, 2)) - C[g][:, o, t].mean(1)
            fz = np.zeros((len(g), 8)); fz[:, 4 * fam:4 * fam + 4] = pr[R[g]]
            rows.append(np.column_stack([C[g, i, t], C[g][:, o, t].mean(1), P[g, i, t], P[g][:, o, t].mean(1), np.full(len(g), fam),
                                         C[g, i, t - 1], P[g, i, :t].sum(1), np.full(len(g), t), np.full(len(g), 23 - t), C[g, i, :t].mean(1), fz, own, oth]))
D = np.concatenate(rows)
for lo, hi in [(0, 0), (1, 40), (41, 1000)]:
    M = D[(D[:, 6] >= lo) & (D[:, 6] <= hi)]
    big = (M[:, 2] >= 10).astype(float); small = ((M[:, 2] >= 1) & (M[:, 2] < 10)).astype(float)
    X = np.column_stack([np.ones(len(M)), M[:, 0], M[:, 1], M[:, 3], M[:, 4], M[:, 5], M[:, 6], M[:, 7], M[:, 9], M[:, 10:18], small, big])
    bo = np.linalg.lstsq(X, M[:, -2], rcond=None)[0][-1]; bt = np.linalg.lstsq(X, M[:, -1], rcond=None)[0][-1]
    pm = M[big > 0, 2].mean(); left = M[:, 8].mean()
    gp = (bo + 3 * bt) / pm
    print(f'earlier punishment {lo}-{hi}: n {len(M)}, n big {int(big.sum())}, mean big dose {pm:.1f}: own {bo:+.2f} others {bt:+.2f} net/pt {0.6 * gp * left - 2:+.2f} (left {left:.1f}), own mean so far {M[:, 9].mean():.1f}')
