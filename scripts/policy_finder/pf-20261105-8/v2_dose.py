"""v2: dose-response of the clones from v1's sweep. Rows: valid player-rounds where
the player was below 16. Regress own c(t+k) and others' mean c(t+k) on p(t) with controls
(c dummies, c(t-1), others' mean, t, number of earlier punishments), separately by
number of earlier punishments and by round group. Rule variation makes p vary given state."""
import numpy as np, sys
z = np.load('/tmp/v1sweep.npz')
C, P, V = z['C'].astype(float), z['P'].astype(float), z['V']
G = C.shape[0]
npun = np.concatenate([np.zeros((G, 4, 1)), np.cumsum(P > 0, 2)[:, :, :-1]], 2)
ptot = np.concatenate([np.zeros((G, 4, 1)), np.cumsum(P, 2)[:, :, :-1]], 2)
rows = []
K = (1, 3, 6, 10)
for t in range(1, 20):
    for i in range(4):
        o = [j for j in range(4) if j != i]
        sel = V[:, i, t] & V[:, i, t - 1] & (C[:, i, t] < 16)
        g = np.where(sel)[0]
        oth = C[g][:, o, t].mean(1)
        rec = [np.full(len(g), t), C[g, i, t], C[g, i, t - 1], oth, P[g, i, t], npun[g, i, t], ptot[g, i, t],
               P[g][:, o, t].mean(1)]
        for k in K:
            tk = min(t + k, 23)
            rec.append(C[g, i, tk]); rec.append(C[g][:, o, tk].mean(1))
        rows.append(np.column_stack(rec))
D = np.concatenate(rows)
cols = ['t', 'c', 'cprev', 'oth', 'p', 'npun', 'ptot', 'pothers'] + sum([[f'own{k}', f'oth{k}'] for k in K], [])
ix = {c: j for j, c in enumerate(cols)}
print('rows', len(D))
def reg(M, label):
    cd = np.stack([(M[:, ix['c']] >= lo) & (M[:, ix['c']] < hi) for lo, hi in [(0, 1), (1, 5), (5, 10), (10, 13), (13, 16)]], 1).astype(float)
    X = np.column_stack([cd, M[:, ix['cprev']], M[:, ix['oth']], M[:, ix['t']], M[:, ix['npun']], M[:, ix['pothers']], M[:, ix['p']]])
    out = []
    for k in K:
        b1 = np.linalg.lstsq(X, M[:, ix[f'own{k}']] - M[:, ix['c']], rcond=None)[0][-1]
        b2 = np.linalg.lstsq(X, M[:, ix[f'oth{k}']] - M[:, ix['oth']], rcond=None)[0][-1]
        out.append(f'k{k}: own {b1:+.3f} oth {b2:+.3f}')
    print(f'{label:28s} n={len(M):7d} meanp={M[:, ix["p"]].mean():5.2f} | ' + ' | '.join(out))
reg(D, 'all')
for lo, hi in [(0, 0), (1, 1), (2, 3), (4, 6), (7, 30)]:
    m = (D[:, ix['npun']] >= lo) & (D[:, ix['npun']] <= hi)
    reg(D[m], f'npun {lo}-{hi}')
for lo, hi in [(1, 2), (3, 5), (6, 9), (10, 14), (15, 19)]:
    m = (D[:, ix['t']] >= lo) & (D[:, ix['t']] <= hi)
    reg(D[m], f't {lo}-{hi}')
for lo, hi in [(0, 1), (1, 5), (5, 10), (10, 13), (13, 16)]:
    m = (D[:, ix['c']] >= lo) & (D[:, ix['c']] < hi)
    reg(D[m], f'c {lo}-{hi}')
# nonlinearity in dose
for lo, hi in [(0, 0), (1, 3), (4, 8), (9, 15), (16, 30)]:
    m = (D[:, ix['p']] >= lo) & (D[:, ix['p']] <= hi)
    print(f'p {lo}-{hi}: n {m.sum()} mean own1 change {(D[m, ix["own1"]] - D[m, ix["c"]]).mean():+.2f}, c {D[m, ix["c"]].mean():.1f}')
