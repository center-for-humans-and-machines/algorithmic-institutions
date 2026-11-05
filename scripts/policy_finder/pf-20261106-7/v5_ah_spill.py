# v5: spillover. Does punishment dealt to the OTHER members of my group move an unpunished member's
# later contribution and exit? cells (t, own c, opp, round(others' mean c), n); regressor: others' total p (and sqrt)
import numpy as np
def fe(y, x, cell):
    _, inv = np.unique(cell, return_inverse=True); cnt = np.bincount(inv)
    def dm(v): return v - (np.bincount(inv, v) / cnt)[inv]
    X = np.stack([dm(c) for c in x], 1); Y = dm(y)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b; s2 = (r**2).sum() / (len(Y) - X.shape[1] - len(cnt))
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
rng = np.random.default_rng(6)
for rounds in [[0], [4], [8, 12]]:
    R = {k: [] for k in 'y1 y8 y2 st st8 op nop c hi cell'.split()}
    for v in ['v1', 'v2', 'v3', 'v4']:
        d = np.load(f'/tmp/{v}sweep.npz'); idx = rng.choice(len(d['C']), 60000, replace=False)
        C, P, G, V = [d[k][idx].astype(np.float32) for k in 'CPGV']; opp = d['opp'][idx]; seat = d['seat'][idx]
        mine = G == seat[:, None, None]
        for t in rounds:
            mv = mine[:, :, t] & (V[:, :, t] > 0)
            s = np.where(mv, C[:, :, t], 0).sum(1, keepdims=True); n = mv.sum(1, keepdims=True)
            om = (s - C[:, :, t]) / np.maximum(n - 1, 1)
            ps = np.where(mine[:, :, t], P[:, :, t], 0).sum(1, keepdims=True)
            op = ps - P[:, :, t]  # punishment dealt to the others
            nop = (np.where(mine[:, :, t], P[:, :, t] > 0, False).sum(1, keepdims=True) - (P[:, :, t] > 0))
            m = mv & (P[:, :, t] == 0) & (n >= 2)
            stay = (G[:, :, t:t + 4] == G[:, :, t:t + 1]).all(2); st8 = (G[:, :, t:t + 9] == G[:, :, t:t + 1]).all(2)
            sw = (G[:, :, t + 4] != G[:, :, t + 3])
            R['y1'].append((C[:, :, t + 3] - C[:, :, t])[m]); R['y8'].append((C[:, :, t + 8] - C[:, :, t])[m])
            R['y2'].append(sw[m].astype(float)); R['st'].append(stay[m]); R['st8'].append(st8[m])
            R['op'].append(np.broadcast_to(op, m.shape)[m]); R['nop'].append(np.broadcast_to(nop, m.shape)[m]); R['c'].append(C[:, :, t][m])
            R['hi'].append((C[:, :, t] >= 10)[m])
            cell = (((t * 25 + C[:, :, t]) * 5 + opp[:, None]) * 25 + np.round(om)) * 9 + n
            R['cell'].append(cell[m])
    R = {k: np.concatenate(v) for k, v in R.items()}
    for lab, sel0 in [('c<10', ~R['hi']), ('c>=10', R['hi'])]:
        X = [R['op'], R['nop']]
        print(rounds, lab, 'n=%d mean others p %.2f' % (sel0.sum(), R['op'][sel0].mean()))
        for yn, y, sel in [('dc(t+3)', R['y1'], R['st']), ('dc(t+8)', R['y8'], R['st8']), ('switch', R['y2'], np.ones_like(R['st']))]:
            s = sel & sel0
            b, se = fe(y[s], [x[s] for x in X], R['cell'][s])
            print('  %-8s per point to others %+.4f(%.4f)  per other punished %+.4f(%.4f)' % (yn, b[0], se[0], b[1], se[1]))
