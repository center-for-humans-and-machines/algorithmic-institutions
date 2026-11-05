# v3: response to MY rule's punishment, identified from the design's random params:
# my group's members only; within cells (round t, exact own c, opponent, group mean c rounded), regress
#  y1 = c(t+4)-c(t) (stayers), y2 = switch at the block's decision, on p and sqrt(p) (cell-demeaned)
import numpy as np, sys
d = np.load(sys.argv[1] if len(sys.argv)>1 else '/tmp/v2sweep.npz')
rng = np.random.default_rng(4); idx = rng.choice(len(d['C']), 80000, replace=False)
C, P, G, V = [d[k][idx].astype(np.float32) for k in 'CPGV']; opp = d['opp'][idx]; seat = d['seat'][idx]
mine = G == seat[:, None, None]
def fe(y, x, cell):
    # demean y and x columns within cells
    _, inv = np.unique(cell, return_inverse=True)
    cnt = np.bincount(inv)
    def dm(v): return v - (np.bincount(inv, v) / cnt)[inv]
    X = np.stack([dm(c) for c in x], 1); Y = dm(y)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b; s2 = (r**2).sum() / (len(Y) - X.shape[1] - len(cnt))
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se
for rounds in [[0], [1, 2], [4], [5, 6], [8, 12], [9, 10, 13, 14], [16], [17, 18], [20]]:
    Y1, Y2, Xp, CELL, ST = [], [], [], [], []
    for t in rounds:
        nd = (t // 4) * 4 + 3
        gm = np.where(mine[:, :, t], C[:, :, t], 0).sum(1) / np.maximum(mine[:, :, t].sum(1), 1)
        m = mine[:, :, t] & (V[:, :, t] > 0) & (C[:, :, t] <= 12)
        stay = G[:, :, min(t + 4, 23)] == G[:, :, t]
        sw = (G[:, :, nd + 1] != G[:, :, nd]) if nd < 20 else np.zeros_like(stay)
        Y1.append((C[:, :, min(t + 4, 23)] - C[:, :, t])[m]); Y2.append(sw[m].astype(float)); ST.append(stay[m])
        Xp.append(P[:, :, t][m])
        cell = ((t * 25 + C[:, :, t]) * 5 + opp[:, None]) * 25 + np.round(gm)[:, None]
        CELL.append(cell[m])
    Y1, Y2, Xp, CELL, ST = map(np.concatenate, (Y1, Y2, Xp, CELL, ST))
    b1, s1 = fe(Y1[ST], [Xp[ST], np.sqrt(Xp[ST])], CELL[ST])
    b2, s2 = fe(Y2, [Xp], CELL)
    print(rounds, f'n={len(Y1)} mean p={Xp.mean():.2f} | dc(t+4): p {b1[0]:+.3f}({s1[0]:.3f}) sqrt {b1[1]:+.3f}({s1[1]:.3f}) | switch per p {b2[0]:+.4f}({s2[0]:.4f})')
