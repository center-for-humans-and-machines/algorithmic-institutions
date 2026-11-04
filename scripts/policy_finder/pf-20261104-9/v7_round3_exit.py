"""v7: what drives leaving at the first decision (after round 3)? Pooled v3/v5/v6 sweeps, my members in
rounds 0-3 (all 4 start in my group). Linear probability model of leaving on own punishment in rounds 0,1,2,
the per-capita pool gap (mine - ah's) in rounds 0,1,2,3, own c in rounds 0-3 and others' mean c in round 3."""
import numpy as np
Xs, ys = [], []
for v in ['v3', 'v5', 'v6']:
    z = np.load(f'/tmp/pf/{v}.npz'); C = z['C'][:, ::2, :, :5].astype(np.float32); P = z['P'][:, ::2, :, :5].astype(np.float32); G = z['G'][:, ::2, :, :5]
    C = C.reshape(-1, 8, 5); P = P.reshape(-1, 8, 5); G = G.reshape(-1, 8, 5)
    mi = (G[:, :, :4] == 0); ot = ~mi
    pc = lambda msk: (1.6 * (C[:, :, :4] * msk).sum(1) - (P[:, :, :4] * msk).sum(1)) / np.maximum(msk.sum(1), 1)  # [ep, 4]
    gap = pc(mi) - pc(ot)
    m0 = mi[:, :, 0]
    leave = (G[:, :, 4] == 1).astype(np.float32)
    oc3 = ((C[:, :, 3] * mi[:, :, 3]).sum(1, keepdims=True) - C[:, :, 3]) / 3
    feats = [P[:, :, 0], P[:, :, 1], P[:, :, 2]] + [np.broadcast_to(gap[:, k:k + 1], m0.shape) for k in range(4)] + [C[:, :, k] for k in range(4)] + [oc3]
    Xs.append(np.stack([f[m0] for f in feats], -1)); ys.append(leave[m0])
X = np.concatenate(Xs); y = np.concatenate(ys)
names = ['p0', 'p1', 'p2', 'gap0', 'gap1', 'gap2', 'gap3', 'c0', 'c1', 'c2', 'c3', 'oc3']
Xa = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(Xa, y, rcond=None)[0]
res = y - Xa @ b; se = np.sqrt(np.diag(np.linalg.inv(Xa.T @ Xa)) * res.var())
print('N', len(y), 'mean leave', y.mean().round(3))
for k, bb, s in zip(['1'] + names, b, se): print(f'{k:5s} {bb:+.4f} ({s:.4f})')
print('means', dict(zip(names, X.mean(0).round(2))))
# non-parametric: leave by p0 bin x p1+p2 bin
p0, p12 = X[:, 0], X[:, 1] + X[:, 2]
for a, bnd in [(0, 1), (1, 15), (15, 31)]:
    print(f'p0 {a}-{bnd}:', ' '.join(f'p12[{lo},{hi}) {y[(p0>=a)&(p0<bnd)&(p12>=lo)&(p12<hi)].mean():.3f}(n{((p0>=a)&(p0<bnd)&(p12>=lo)&(p12<hi)).sum()})' for lo, hi in [(0, 1), (1, 15), (15, 40), (40, 61)]))
