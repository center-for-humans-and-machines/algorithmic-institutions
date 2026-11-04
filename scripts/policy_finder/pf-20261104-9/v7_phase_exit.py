"""v7: per decision (after rounds 3, 7, 11), linear probability of leaving my group for members present the
whole phase, on own punishment at each phase position 0-2 and the per-capita pool gap (mine - ah's) at each
position 0-3, plus own c and others' c at the last round. Also joining from ah's group on the same gaps.
Pooled v3/v5/v6 sweeps, every other episode."""
import numpy as np
def fit(X, y, names, title):
    Xa = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(Xa, y, rcond=None)[0]
    res = y - Xa @ b; se = np.sqrt(np.diag(np.linalg.inv(Xa.T @ Xa)) * res.var())
    print(title, 'N', len(y), 'mean', y.mean().round(3), ' '.join(f'{k}={bb:+.4f}({s:.4f})' for k, bb, s in zip(['1'] + names, b, se)))
for r0 in [0, 4, 8]:
    Xs, ys, Xj, yj = [], [], [], []
    for v in ['v3', 'v5', 'v6']:
        z = np.load(f'/tmp/pf/{v}.npz'); sl = slice(r0, r0 + 5)
        C = z['C'][:, ::2, :, sl].astype(np.float32).reshape(-1, 8, 5); P = z['P'][:, ::2, :, sl].astype(np.float32).reshape(-1, 8, 5); G = z['G'][:, ::2, :, sl].reshape(-1, 8, 5)
        mi = (G[:, :, :4] == 0); ot = ~mi
        pc = lambda msk: (1.6 * (C[:, :, :4] * msk).sum(1) - (P[:, :, :4] * msk).sum(1)) / np.maximum(msk.sum(1), 1)
        gap = pc(mi) - pc(ot)
        ok = (mi[:, :, 0:1] == mi[:, :, :4]).all(-1)   # same group all phase (always true within a phase)
        nm = mi[:, :, 0].sum(1); no = 8 - nm
        good = (nm >= 2) & (no >= 1)
        stay_in = mi[:, :, 0] & good[:, None]; join_c = ot[:, :, 0] & good[:, None]
        oc3 = ((C[:, :, 3] * mi[:, :, 3]).sum(1, keepdims=True) - C[:, :, 3] * mi[:, :, 3]) / np.maximum(nm[:, None] - 1, 1)
        gapb = [np.broadcast_to(gap[:, k:k + 1], (len(C), 8)) for k in range(4)]
        f = [P[:, :, 0], P[:, :, 1], P[:, :, 2]] + gapb + [C[:, :, 3], oc3]
        Xs.append(np.stack([a[stay_in] for a in f], -1)); ys.append((G[:, :, 4] == 1)[stay_in])
        fj = gapb + [C[:, :, 3]]
        Xj.append(np.stack([a[join_c] for a in fj], -1)); yj.append((G[:, :, 4] == 0)[join_c])
    fit(np.concatenate(Xs), np.concatenate(ys).astype(float), ['p0', 'p1', 'p2', 'g0', 'g1', 'g2', 'g3', 'c3', 'oc3'], f'leave after {r0+3}:')
    fit(np.concatenate(Xj), np.concatenate(yj).astype(float), ['g0', 'g1', 'g2', 'g3', 'c3'], f'join  after {r0+3}:')
