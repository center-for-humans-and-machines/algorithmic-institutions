"""v4: true score of each of the 768 sweep rules (v1-v3) regressed on its punishment spend per
player per game in cells round-bucket x recipient contribution class (one unit = 4 points)."""
import numpy as np
rb = [(0, 1), (1, 3), (3, 5), (5, 8), (8, 11), (11, 14), (14, 18), (18, 24)]
cb = [(0, 1), (1, 5), (5, 10), (10, 14), (14, 17), (17, 21)]
X, y, fam = [], [], []
for k, f in enumerate(['/tmp/v1sweep.npz', '/tmp/v2sweep.npz', '/tmp/v3sweep.npz']):
    z = np.load(f); C, P, V, R, pool = z['C'], z['P'].astype(float), z['V'], z['R'], z['pool']
    P = P * V
    nr = R.max() + 1
    feats = []
    for (a, b) in rb:
        for (lo, hi) in cb:
            m = (C[:, :, a:b] >= lo) & (C[:, :, a:b] < hi)
            s = (P[:, :, a:b] * m).sum((1, 2)) / 4 / 4   # units per player per game
            feats.append(np.bincount(R, s, nr) / np.bincount(R, None, nr))
    X.append(np.array(feats).T); y.append(pool); fam += [k] * nr
X = np.vstack(X); y = np.concatenate(y); fam = np.array(fam)
D = np.column_stack([np.ones(len(y)), fam == 1, fam == 2, X])
beta, *_ = np.linalg.lstsq(D, y, rcond=None)
res = y - D @ beta
print('n', len(y), 'R2', 1 - res.var() / y.var(), 'resid sd', res.std())
# bootstrap se
B = []
rng = np.random.default_rng(0)
for _ in range(200):
    i = rng.integers(0, len(y), len(y)); b, *_ = np.linalg.lstsq(D[i], y[i], rcond=None); B.append(b)
se = np.std(B, 0)
k = 3
print('payoff per unit (4 pts): rows rounds, cols c classes', cb)
for (a, b) in rb:
    print(f'rounds {a:2d}-{b-1:2d}', ' '.join(f'{beta[k+j]:+6.1f}({se[k+j]:4.1f})' for j in range(len(cb))),
          '| mean spend', ' '.join(f'{X[:, k-3+j].mean():.2f}' for j in range(len(cb))))
    k += len(cb)
