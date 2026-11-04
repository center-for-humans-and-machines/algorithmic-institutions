"""v6: does a laggard's history (own past mean vs this round) change the value of punishing it?
Pooled sweeps (from /tmp/pf/<ver>.npz, see sweep_load.py). Rows: my member, valid, rounds 1,2,4,5,6,8,9,10,
c < 0.9 x others' mean. Outcome: c three rounds later (members still in my group and valid), and staying in
my group at the next decision. OLS per history class: y ~ p + c + o + hist + round dummies."""
import numpy as np, sys
vers = sys.argv[1:] or ['pf-20261104-9-v5', 'pf-20261104-9-v4', 'pf-20261104-9-v3']
rows = []
for v in vers:
    z = np.load(f'/tmp/pf/{v}.npz')
    C = z['C'][:, ::2].astype(np.float32); V = z['V'][:, ::2].astype(np.float32)
    P = z['P'][:, ::2].astype(np.float32); G = z['G'][:, ::2]
    mine = (G == 0).astype(np.float32)
    for t in [1, 2, 4, 5, 6, 8, 9, 10]:
        m = mine[..., t] * V[..., t]
        tot = (C[..., t] * m).sum(-1, keepdims=True); n = m.sum(-1, keepdims=True)
        o = (tot - C[..., t] * m) / np.maximum(n - m, 1)
        hist = C[..., :t].sum(-1) / t  # own past mean, all rounds (any group)
        nd = (t // 4) * 4 + 4  # first round after next decision
        stay = mine[..., nd]
        t3 = t + 3
        ok3 = mine[..., t3] * V[..., t3]
        sel = (m > 0) & (n - m > 0) & (C[..., t] < 0.9 * o)
        idx = np.nonzero(sel)
        rows.append(np.stack([np.full(len(idx[0]), t), C[..., t][idx], o[idx], hist[idx], P[..., t][idx],
                              C[..., t3][idx], ok3[idx], stay[idx]], 1))
R = np.concatenate(rows).astype(np.float64)
t, c, o, h, p, c3, ok3, st = R.T
print('rows', len(R))
def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return b
cls = {'dip (hist >= c+4)': h - c >= 4, 'steady': np.abs(h - c) < 4, 'rising (hist <= c-4)': h - c <= -4,
       'chronic low (hist < 0.6 o)': h < 0.6 * o, 'hist >= 0.6 o': h >= 0.6 * o,
       'c==0': c == 0, 'c>0': c > 0}
for name, s in cls.items():
    D = np.stack([(t == k) for k in [1, 2, 4, 5, 6, 8, 9, 10]], 1).astype(float)
    X = np.column_stack([p, c, o, h, D])
    s3 = s & (ok3 > 0)
    b3 = ols(c3[s3], X[s3]); bs = ols(st[s], X[s])
    print(f'{name:28s} n={s.sum():8d} mean p={p[s].mean():5.1f} lift/pt={b3[0]:+.4f} stay/pt={bs[0]:+.5f} '
          f'ratio={b3[0] / max(-bs[0], 1e-6):6.1f} stay={st[s].mean():.2f}')
