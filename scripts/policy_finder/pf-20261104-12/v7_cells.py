"""Pool the v3-v6 sweeps (1024 design points) on a common feature space: per point,
the mean punishment dealt per member-round in each cell (round group x contribution bin),
as the conditional mean dose (the rule's intensity there). Regress the point's pool on
them (OLS, then ridge-ish check) to read the marginal value of a point of intensity per cell."""
import numpy as np
cells_r = {'b0p0': [0], 'b0p1': [1], 'b1p0': [4], 'b1p1': [5], 'b2p0': [8], 'b2p1': [9], 'b34p0': [12, 16], 'b34p1': [13, 17]}
cbins = [(0, 4), (5, 9), (10, 20)]
F, Y, V, N = [], [], [], []
for v in ['v3', 'v4', 'v5', 'v6']:
    z = np.load(f'/tmp/pf-20261104-12-{v}.npz')
    c, p, g = z['c'], z['p'].astype(np.float32), z['g']
    our = g == 0
    feats = []
    for rn, rs in cells_r.items():
        for lo, hi in cbins:
            m = our[..., rs] & (c[..., rs] >= lo) & (c[..., rs] <= hi)
            dose = (p[..., rs] * m).sum((1, 2, 3)) / np.maximum(m.sum((1, 2, 3)), 1)
            feats.append(dose)
    F.append(np.stack(feats, 1)); Y.append(z['pool']); V += [v] * len(z['pool'])
    # members
    N.append(our.sum(2).mean((1, 2)))
F = np.concatenate(F); Y = np.concatenate(Y); V = np.array(V); N = np.concatenate(N)
names = [f'{r}_{lo}-{hi}' for r in cells_r for lo, hi in cbins]
print('pool mean by version', {v: round(Y[V == v].mean(), 2) for v in np.unique(V)})
Fs = (F - F.mean(0)) / F.std(0)
Z = np.column_stack([np.ones(len(Y)), Fs] + [(V == v).astype(float) for v in ['v4', 'v5', 'v6']])
b, res, *_ = np.linalg.lstsq(Z, Y, rcond=None)
r = Y - Z @ b; s2 = r.var() * len(Y) / (len(Y) - Z.shape[1])
se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * s2)
print('R2', 1 - r.var() / Y.var(), 'resid sd', r.std())
for i, n in enumerate(names):
    print(f'{n:14s} mean dose {F[:, i].mean():5.1f} sd {F[:, i].std():5.1f}  per sd {b[1+i]:+6.2f} ({se[1+i]:.2f})  per point {b[1+i]/F[:, i].std():+.3f}')
# top points
o = np.argsort(-Y)[:15]
print('top points: pool, version, doses for 0-4 at b0p0,b0p1,b1p0,b1p1,b2p0 and 5-9 at b0p0,b0p1')
for i in o:
    print(f'{Y[i]:6.1f} {V[i]} n={N[i]:.2f} ' + ' '.join(f'{F[i, j]:5.1f}' for j in [0, 3, 6, 9, 12, 1, 4, 7, 10]))
np.savez('/tmp/v7_cells.npz', F=F, Y=Y, V=V, N=N, names=names)
