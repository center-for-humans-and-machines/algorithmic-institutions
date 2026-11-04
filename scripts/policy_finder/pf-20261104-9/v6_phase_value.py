"""v6: value of a point of punishment to a laggard by phase (0: rounds 0-2, 1: 4-6, 2: 8-10) and position.
Pooled v3-v5 sweeps. Rows: my member, valid, c < 0.9 x others' mean. Outcomes: c in round 13 for members
in my group then (long-run lift), staying in my group past the next decision, and still in my group in round 13.
OLS y ~ p + c + o + round dummies, per phase and position (0 vs 1-2)."""
import numpy as np
vers = ['pf-20261104-9-v5', 'pf-20261104-9-v4', 'pf-20261104-9-v3']
rows = []
for v in vers:
    z = np.load(f'/tmp/pf/{v}.npz')
    C = z['C'][:, ::2].astype(np.float32); V = z['V'][:, ::2].astype(np.float32)
    P = z['P'][:, ::2].astype(np.float32); G = z['G'][:, ::2]
    mine = (G == 0).astype(np.float32)
    for t in [0, 1, 2, 4, 5, 6, 8, 9, 10]:
        m = mine[..., t] * V[..., t]
        tot = (C[..., t] * m).sum(-1, keepdims=True); n = m.sum(-1, keepdims=True)
        o = (tot - C[..., t] * m) / np.maximum(n - m, 1)
        nd = (t // 4) * 4 + 4
        sel = (m > 0) & (n - m > 0) & (C[..., t] < 0.9 * o)
        idx = np.nonzero(sel)
        rows.append(np.stack([np.full(len(idx[0]), t), C[..., t][idx], o[idx], P[..., t][idx],
                              C[..., 13][idx], (mine[..., 13] * V[..., 13])[idx], mine[..., nd][idx]], 1))
R = np.concatenate(rows).astype(np.float64)
t, c, o, p, c13, in13, st = R.T
def ols(y, X): return np.linalg.lstsq(X, y, rcond=None)[0]
for ph in range(3):
    for pos, ps in [('0', [0]), ('1-2', [1, 2])]:
        rs = [4 * ph + q for q in ps]
        s = np.isin(t, rs)
        D = np.stack([(t == k) for k in rs], 1).astype(float)
        X = np.column_stack([p, c, o, D])
        a = s & (in13 > 0)
        bl = ols(c13[a], X[a]); bs = ols(st[s], X[s]); bi = ols(in13[s], X[s])
        # pool value per point over rounds to 13..23 approx: 1.6 x (lift x 11 rounds x stay13 - P(in13 loss) x c13 x 11)
        val = 1.6 * 11 * (bl[0] * in13[s].mean() + bi[0] * c13[a].mean())
        print(f'phase {ph} pos {pos:3s} n={s.sum():8d} mean p={p[s].mean():5.1f} lift13/pt={bl[0]:+.4f} '
              f'stay/pt={bs[0]:+.5f} in13/pt={bi[0]:+.5f} late value/pt={val:+.2f} (cost 1)')
