"""v7: herding vs concentration at round 0. Pools v3/v5/v6 sweeps (/tmp/pf/<v>.npz).
For unpunished members in round 0: OLS of c in rounds 1,3,8 and of staying after round 3
on own c0, others' mean c0, others' total p0, and the number of punished others.
For punished members: stay rate and c lift by dose."""
import numpy as np
rows = []
for v in ['v3', 'v5', 'v6']:
    z = np.load(f'/tmp/pf/{v}.npz'); C = z['C'][:, ::2].astype(np.float32); V = z['V'][:, ::2]; P = z['P'][:, ::2].astype(np.float32); G = z['G'][:, ::2]
    m = (G[..., 0] == 0) & (V[..., 0] == 1)  # [pt, ep, 8]
    c0 = C[..., 0]; p0 = P[..., 0]
    n = m.sum(-1, keepdims=True)
    tc = (c0 * m).sum(-1, keepdims=True); tp = (p0 * m).sum(-1, keepdims=True)
    npun = ((p0 > 0) & m).sum(-1, keepdims=True)
    oc = (tc - c0) / np.maximum(n - 1, 1); op = tp - p0; onp = npun - (p0 > 0)
    stay = (G[..., 4] == 0).astype(np.float32)
    rows.append(np.stack([np.broadcast_to(a, c0.shape)[m] for a in
        [c0, p0, oc, op, onp, C[..., 1], C[..., 3], C[..., 8], stay, G[..., 1] == 0, n]], -1))
R = np.concatenate(rows); c0, p0, oc, op, onp, c1, c3, c8, stay, st1, n = R.T
print('rows', len(R))
u = (p0 == 0) & (n == 4)
def ols(y, X, names, mask):
    X = np.column_stack([np.ones(mask.sum())] + [x[mask] for x in X]); b = np.linalg.lstsq(X, y[mask], rcond=None)[0]
    res = y[mask] - X @ b; se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var())
    return ' '.join(f'{k}={bb:+.4f}({s:.4f})' for k, bb, s in zip(['1'] + names, b, se))
for nm, y in [('c1', c1), ('c3', c3), ('c8*stay', c8 * stay), ('stay', stay)]:
    print('unpunished', nm, ols(y, [c0, oc, op, onp], ['c0', 'oc', 'op', 'n_pun_oth'], u))
# concentration: same total others' punishment, 1 punished vs 2+
for lo, hi in [(20, 35), (35, 50), (50, 70)]:
    for k in [1, 2, 3]:
        s = u & (op >= lo) & (op < hi) & (onp == k)
        if s.sum() > 200:
            print(f'op {lo}-{hi} n_pun {k}: N={s.sum()} c0={c0[s].mean():.2f} oc={oc[s].mean():.2f} dc3={(c3-c0)[s].mean():+.2f} dc8={(c8-c0)[s & (stay>0)].mean():+.2f} stay={stay[s].mean():.3f}')
# punished member: by dose, laggards c0 < 0.8 oc
lag = (c0 < 0.8 * oc) & (n == 4)
for lo, hi in [(0, 1), (1, 10), (10, 20), (20, 29), (29, 31)]:
    s = lag & (p0 >= lo) & (p0 < hi)
    print(f'laggard p0 {lo}-{hi}: N={s.sum()} c0={c0[s].mean():.2f} oc={oc[s].mean():.2f} dc3={(c3-c0)[s].mean():+.2f} dc8|stay={(c8-c0)[s & (stay>0)].mean():+.2f} stay={stay[s].mean():.3f}')
