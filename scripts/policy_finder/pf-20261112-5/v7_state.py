"""v7: value of a round-R shock by visible state, from /tmp/pf/pairs.npz (v7_pairs.py). Pairs oriented so the
dose diff is >0; d lead (R..23) ~ dose/10 x state bin + later points diff (per 10, 3 phases). Clustered-free OLS."""
import numpy as np, sys
z = np.load('/tmp/pf/pairs.npz')
R0 = int(sys.argv[1]) if len(sys.argv) > 1 else 4
i = (z['R'] == R0) & (z['dd'] != 0)
s = np.sign(z['dd'][i]).astype(float)
dd = z['dd'][i] * s / 10; late = z['late'][i] * s[:, None] / 10
Y = {k: z[k][i] * s for k in ['dlead', 'down', 'doth', 'dmem']}
st = {k: z[k][i] for k in ['n', 'no', 'new', 'cm', 'ntar']}
print('R', R0, 'pairs', i.sum(), 'mean dose diff', (dd * 10).mean().round(1))
def fit(bins, label):
    cols = [dd * b for b in bins.values()] + [late[:, j] for j in range(3)] + [b * 1.0 for b in bins.values()]
    X = np.stack(cols, 1)
    print('--', label)
    for k, y in Y.items():
        b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
        s2 = (r ** 2).sum() / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
        print(f'  {k:6s}', ' '.join(f'{n}:{b[j]:+.2f}({se[j]:.2f})' for j, n in enumerate(bins)), '| late', ' '.join(f'{b[len(bins)+j]:+.2f}' for j in range(3)))
    print('  n', {n: int(b.sum()) for n, b in bins.items()})
fit({'all': np.ones_like(dd, bool)}, 'all')
n = st['n']; fit({f'n{a}': (n == a) if a < 6 else (n >= 6) for a in [2, 3, 4, 5, 6]}, 'my size')
fit({f'new{a}': (st['new'] == a) if a < 3 else (st['new'] >= 3) for a in [0, 1, 2, 3]}, 'newcomers')
cm = st['cm']; fit({'cm<6': cm < 6, 'cm6-9': (cm >= 6) & (cm < 9), 'cm9-12': (cm >= 9) & (cm < 12), 'cm12+': cm >= 12}, 'my mean c')
fit({f'tar{a}': (st['ntar'] == a) if a < 3 else (st['ntar'] >= 3) for a in [1, 2, 3]}, 'n targets differing')
