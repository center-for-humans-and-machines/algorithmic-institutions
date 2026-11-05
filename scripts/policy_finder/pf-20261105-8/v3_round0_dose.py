"""v3: round-0 punishment of a player, by own round-0 contribution class, in dose bins (dummies), pooled
over both sweeps; outcome: own c and others' mean c averaged over rounds 1-10 minus round 0.
Controls: own c0, others' mean c0, others' mean p0, family dummy x that family's params."""
import numpy as np
rows = []
for fam, f in enumerate(('/tmp/v1sweep.npz', '/tmp/v2sweep.npz')):
    z = np.load(f); C, P, V, R, prm = z['C'].astype(np.float32), z['P'].astype(np.float32), z['V'], z['R'], z['prm']
    if fam == 0: pr = np.column_stack([prm[:, 0], np.log(prm[:, 1]), np.minimum(prm[:, 2], 13), np.log(prm[:, 3])])
    else: pr = np.column_stack([prm[:, 0], prm[:, 1], np.log(prm[:, 2]), np.minimum(prm[:, 3], 13)])
    for i in range(4):
        o = [j for j in range(4) if j != i]
        g = np.where(V[:, :, 0].all(1) & V[:, i, 1:11].all(1))[0]
        own = C[g, i, 1:11].mean(1) - C[g, i, 0]; oth = C[g][:, o, 1:11].mean((1, 2)) - C[g][:, o, 0].mean(1)
        pp = pr[R[g]]
        fz = np.zeros((len(g), 8)); fz[:, 4 * fam:4 * fam + 4] = pp
        rows.append(np.column_stack([C[g, i, 0], C[g][:, o, 0].mean(1), P[g, i, 0], P[g][:, o, 0].mean(1), np.full(len(g), fam), fz, own, oth]))
D = np.concatenate(rows)
PB = [(0, 0), (1, 3), (4, 7), (8, 12), (13, 19), (20, 30)]
for lo, hi in [(0, 3), (3, 8), (8, 12), (12, 16), (16, 19), (19, 21)]:
    m = (D[:, 0] >= lo) & (D[:, 0] < hi); M = D[m]
    pd_ = np.stack([(M[:, 2] >= a) & (M[:, 2] <= b) for a, b in PB[1:]], 1).astype(float)
    X = np.column_stack([np.ones(len(M)), M[:, 0], M[:, 1], M[:, 3], M[:, 4], M[:, 5:13], pd_])
    bo = np.linalg.lstsq(X, M[:, -2], rcond=None)[0][-len(PB) + 1:]
    bt = np.linalg.lstsq(X, M[:, -1], rcond=None)[0][-len(PB) + 1:]
    cnt = [int(((M[:, 2] >= a) & (M[:, 2] <= b)).sum()) for a, b in PB]
    print(f'c0 {lo}-{hi}: n per p-bin {cnt}')
    for (a, b), x, y in zip(PB[1:], bo, bt):
        mid = (a + b) / 2
        print(f'   p {a:2d}-{b:2d}: own {x:+.2f} others {y:+.2f}  group per point {(x + 3 * y) / mid:+.3f}')
