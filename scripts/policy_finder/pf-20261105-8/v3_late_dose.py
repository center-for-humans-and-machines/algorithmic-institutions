"""v3: like v3_round0_dose.py but for rounds t in [a, b] (args): punishment at t in dose bins,
by own c(t) class; outcome own and others' mean c over rounds t+1..min(t+W,23) minus round t.
Also prints the implied net payoff per point: 0.6 * group-per-point * rounds left - 2."""
import numpy as np, sys
a, b = int(sys.argv[1]), int(sys.argv[2]); W = int(sys.argv[3]) if len(sys.argv) > 3 else 23
rows = []
for fam, f in enumerate(('/tmp/v1sweep.npz', '/tmp/v2sweep.npz')):
    z = np.load(f); C, P, V, R, prm = z['C'].astype(np.float32), z['P'].astype(np.float32), z['V'], z['R'], z['prm']
    if fam == 0: pr = np.column_stack([prm[:, 0], np.log(prm[:, 1]), np.minimum(prm[:, 2], 24), np.log(prm[:, 3])])
    else: pr = np.column_stack([prm[:, 0], prm[:, 1], np.log(prm[:, 2]), np.minimum(prm[:, 3], 24)])
    for t in range(a, b + 1):
        e = min(t + W, 23)
        for i in range(4):
            o = [j for j in range(4) if j != i]
            g = np.where(V[:, :, t].all(1) & V[:, i, t + 1:e + 1].all(1))[0]
            own = C[g, i, t + 1:e + 1].mean(1) - C[g, i, t]; oth = C[g][:, o, t + 1:e + 1].mean((1, 2)) - C[g][:, o, t].mean(1)
            fz = np.zeros((len(g), 8)); fz[:, 4 * fam:4 * fam + 4] = pr[R[g]]
            rows.append(np.column_stack([C[g, i, t], C[g][:, o, t].mean(1), P[g, i, t], P[g][:, o, t].mean(1), np.full(len(g), fam),
                                         C[g, i, t - 1], P[g, i, :t].sum(1), np.full(len(g), t), np.full(len(g), e - t), fz, own, oth]))
D = np.concatenate(rows)
PB = [(0, 0), (1, 3), (4, 7), (8, 12), (13, 19), (20, 30)]
for lo, hi in [(0, 3), (3, 8), (8, 12), (12, 16), (16, 21)]:
    m = (D[:, 0] >= lo) & (D[:, 0] < hi); M = D[m]
    pd_ = np.stack([(M[:, 2] >= x) & (M[:, 2] <= y) for x, y in PB[1:]], 1).astype(float)
    X = np.column_stack([np.ones(len(M)), M[:, 0], M[:, 1], M[:, 3], M[:, 4], M[:, 5], M[:, 6], M[:, 7], M[:, 9:17], pd_])
    bo = np.linalg.lstsq(X, M[:, -2], rcond=None)[0][-5:]; bt = np.linalg.lstsq(X, M[:, -1], rcond=None)[0][-5:]
    cnt = [int(((M[:, 2] >= x) & (M[:, 2] <= y)).sum()) for x, y in PB]; left = M[:, 8].mean()
    print(f't {a}-{b} c {lo}-{hi}: n per p-bin {cnt}, rounds left {left:.1f}')
    for (x, y), u, v in zip(PB[1:], bo, bt):
        if cnt[PB.index((x, y))] < 300: continue
        mid = (x + y) / 2; gp = (u + 3 * v) / mid
        print(f'   p {x:2d}-{y:2d}: own {u:+.2f} others {v:+.2f} group/pt {gp:+.3f} net payoff/pt {0.6 * gp * left - 2:+.2f}')
