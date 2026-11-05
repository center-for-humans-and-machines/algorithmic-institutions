# v5: does the lift / exit per sqrt point of round-t punishment depend on the gap between the player's
# contribution and the rest of my group (others' mean), at fixed own contribution? all v1-v4 sweeps pooled (sample)
import numpy as np, sys
K = 3
def fe(y, x, cell):
    _, inv = np.unique(cell, return_inverse=True); cnt = np.bincount(inv)
    def dm(v): return v - (np.bincount(inv, v) / cnt)[inv]
    X = np.stack([dm(c) for c in x], 1); Y = dm(y)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b; s2 = (r**2).sum() / (len(Y) - X.shape[1] - len(cnt))
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
rng = np.random.default_rng(5)
for rounds in [[0], [4], [8, 12]]:
    R = {k: [] for k in 'y1 y8 y2 st st8 p gap c n cell'.split()}
    for v in ['v1', 'v2', 'v3', 'v4']:
        d = np.load(f'/tmp/{v}sweep.npz'); idx = rng.choice(len(d['C']), 60000, replace=False)
        C, P, G, V = [d[k][idx].astype(np.float32) for k in 'CPGV']; opp = d['opp'][idx]; seat = d['seat'][idx]
        mine = G == seat[:, None, None]
        for t in rounds:
            nd = t + 3
            mv = mine[:, :, t] & (V[:, :, t] > 0)
            s = np.where(mv, C[:, :, t], 0).sum(1, keepdims=True); n = mv.sum(1, keepdims=True)
            om = (s - C[:, :, t]) / np.maximum(n - 1, 1)  # others' mean
            m = mv & (C[:, :, t] <= 12) & (n >= 2)
            stay = (G[:, :, t:t + K + 1] == G[:, :, t:t + 1]).all(2)
            st8 = (G[:, :, t:t + 9] == G[:, :, t:t + 1]).all(2)
            sw = (G[:, :, nd + 1] != G[:, :, nd])
            R['y1'].append((C[:, :, t + K] - C[:, :, t])[m]); R['y8'].append((C[:, :, t + 8] - C[:, :, t])[m])
            R['y2'].append(sw[m].astype(float)); R['st'].append(stay[m]); R['st8'].append(st8[m])
            R['p'].append(P[:, :, t][m]); R['gap'].append((om - C[:, :, t])[m]); R['c'].append(C[:, :, t][m]); R['n'].append(np.broadcast_to(n, m.shape)[m])
            cell = (((t * 25 + C[:, :, t]) * 5 + opp[:, None]) * 25 + np.round(om)) * 9 + n
            R['cell'].append(cell[m])
    R = {k: np.concatenate(v) for k, v in R.items()}
    sq = np.sqrt(R['p']); gap = R['gap']; g0 = gap - gap.mean()
    X = [sq, sq * g0, sq * (R['n'] - 4)]
    names = ['sqrt p', 'sqrt p * gap', 'sqrt p * (n-4)']
    print(rounds, 'n=%d mean gap %.2f sd %.2f mean p %.2f' % (len(sq), gap.mean(), gap.std(), R['p'].mean()))
    for yn, y, sel in [('dc(t+3)', R['y1'], R['st']), ('dc(t+8)', R['y8'], R['st8']), ('switch', R['y2'], np.ones_like(R['st']))]:
        b, s = fe(y[sel], [x[sel] for x in X], R['cell'][sel])
        print('  %-8s ' % yn + ' '.join(f'{nm} {bb:+.4f}({ss:.4f})' for nm, bb, ss in zip(names, b, s)))
