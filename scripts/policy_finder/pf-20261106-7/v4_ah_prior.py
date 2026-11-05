# v4: lift and exit per sqrt point of my block-start punishment, by punishment already received (either manager)
# my group's valid members at rounds 4, 8, 12, 16 with c <= 12; within cells (t, c, opp, group mean, new, ever-punished)
# y1 = c(t+k) - c(t) for players still in my group at t+k; y2 = switch at the block's decision
import numpy as np, sys
d = np.load(sys.argv[1] if len(sys.argv)>1 else '/tmp/v3sweep.npz')
K = int(sys.argv[2]) if len(sys.argv)>2 else 3
rng = np.random.default_rng(5); idx = rng.choice(len(d['C']), 100000, replace=False)
C, P, G, V = [d[k][idx].astype(np.float32) for k in 'CPGV']; opp = d['opp'][idx]; seat = d['seat'][idx]
mine = G == seat[:, None, None]
def fe(y, x, cell):
    _, inv = np.unique(cell, return_inverse=True); cnt = np.bincount(inv)
    def dm(v): return v - (np.bincount(inv, v) / cnt)[inv]
    X = np.stack([dm(c) for c in x], 1); Y = dm(y)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b; s2 = (r**2).sum() / (len(Y) - X.shape[1] - len(cnt))
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
for rounds in [[4], [8, 12], [16]]:
    R = {k: [] for k in 'y1 y2 st p new ever prior cell'.split()}
    for t in rounds:
        nd = t + 3
        gm = np.where(mine[:, :, t], C[:, :, t], 0).sum(1) / np.maximum(mine[:, :, t].sum(1), 1)
        m = mine[:, :, t] & (V[:, :, t] > 0) & (C[:, :, t] <= 12)
        new = mine[:, :, t] & ~mine[:, :, t - 1]
        prior = P[:, :, :t].sum(2)  # punishment received so far from either manager
        ever = prior > 0
        tk = min(t + K, 23)
        stay = (G[:, :, t:tk + 1] == G[:, :, t:t + 1]).all(2)
        sw = (G[:, :, nd + 1] != G[:, :, nd]) if nd < 20 else np.zeros_like(stay)
        R['y1'].append((C[:, :, tk] - C[:, :, t])[m]); R['y2'].append(sw[m].astype(float)); R['st'].append(stay[m])
        R['p'].append(P[:, :, t][m]); R['new'].append(new[m].astype(float)); R['ever'].append(ever[m].astype(float)); R['prior'].append(prior[m])
        cell = ((((t * 25 + C[:, :, t]) * 5 + opp[:, None]) * 25 + np.round(gm)[:, None]) * 2 + new) * 2 + ever
        R['cell'].append(cell[m])
    R = {k: np.concatenate(v) for k, v in R.items()}
    st = R['st']; p = R['p']; sq = np.sqrt(p); new = R['new']; nev = 1 - R['ever']
    pr = R["prior"]; names = ["sqrt(prior0)", "sqrt(1-15)", "sqrt(15-40)", "sqrt(40+)"]
    X = [sq*(pr==0), sq*((pr>0)&(pr<=15)), sq*((pr>15)&(pr<=40)), sq*(pr>40)]
    b1, s1 = fe(R['y1'][st], [x[st] for x in X], R['cell'][st])
    b2, s2 = fe(R['y2'], X, R['cell'])
    print(rounds, f'n={len(p)} share new {new.mean():.2f} never {nev.mean():.2f} new&never {(new*nev).mean():.2f}, mean p {p.mean():.2f}')
    print('  dc(t+%d): ' % K + ' '.join(f'{n} {b:+.3f}({s:.3f})' for n, b, s in zip(names, b1, s1)))
    print('  switch : ' + ' '.join(f'{n} {b:+.4f}({s:.4f})' for n, b, s in zip(names, b2, s2)))
