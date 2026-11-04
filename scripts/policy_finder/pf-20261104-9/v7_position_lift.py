"""v7: lift per point by phase position. Pooled v3/v5/v6 sweeps. For my members present in round r
(r = 0,1,2 and 4,5,6), laggards (c_r < 0.9 x others' mean): OLS of c at the phase's last round+1 (r_dec+1, if
still in my group) and of c at round r+1, on own p_r, with controls c_r, c_{r-1}, own p_{r-1}, others' mean
c_r and others' total p_r."""
import numpy as np
D = {}
for v in ['v3', 'v5', 'v6']:
    z = np.load(f'/tmp/pf/{v}.npz')
    D[v] = [z[k][:, ::2].astype(np.float32).reshape(-1, 8, 24) for k in 'CVPG']
for r in [0, 1, 2, 4, 5, 6, 8, 9, 10]:
    rows = []
    for v in D:
        C, V, P, G = D[v]
        m = (G[..., r] == 0) & (V[..., r] == 1); n = m.sum(-1, keepdims=True)
        c = C[..., r]; oc = ((c * m).sum(-1, keepdims=True) - c) / np.maximum(n - 1, 1)
        op = (P[..., r] * m).sum(-1, keepdims=True) - P[..., r]
        lag = m & (c < 0.9 * oc) & (n >= 3)
        dec = (r // 4) * 4 + 3
        cl = C[..., r - 1] if r > 0 else c; pl = P[..., r - 1] if r > 0 else 0 * c
        tgt = dec + 5
        inn = (G[..., tgt] == 0)
        f = [C[..., r + 1], C[..., tgt], inn.astype(np.float32), P[..., r], c, cl, pl, oc, op]
        rows.append(np.stack([np.broadcast_to(a, c.shape)[lag] for a in f], -1))
    R = np.concatenate(rows)
    y1, y2, st, X = R[:, 0], R[:, 1], R[:, 2], R[:, 3:]
    out = []
    for y, msk, nm in [(y1, np.ones(len(R), bool), 'c[r+1]'), (y2, st > 0, f'c[{(r//4)*4+8}]|in'), (st, np.ones(len(R), bool), 'in@+dec')]:
        Xa = np.column_stack([np.ones(msk.sum()), X[msk]]); b = np.linalg.lstsq(Xa, y[msk], rcond=None)[0]
        out.append(f'{nm} {b[1]:+.4f}')
    print(f'r={r:2d} N={len(R)} mean p={X[:,0].mean():.1f}  per point: ' + '  '.join(out))
