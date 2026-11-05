"""v3: across the 512 rules of v1's and v2's sweeps, regress each rule's true score on its
punishment spend (per player per game) in cells of (round bin x recipient's contribution bin)."""
import numpy as np
TB = [(0, 1), (1, 4), (4, 9), (9, 14), (14, 24)]
CB = [(0, 3), (3, 8), (8, 13), (13, 21)]
X, y, src = [], [], []
for f in ('/tmp/v1sweep.npz', '/tmp/v2sweep.npz'):
    z = np.load(f); C, P, R = z['C'].astype(float), z['P'].astype(float), z['R']
    nr = len(z['pool'])
    feats = []
    for (a, b) in TB:
        for (lo, hi) in CB:
            m = (C[:, :, a:b] >= lo) & (C[:, :, a:b] < hi)
            s = (P[:, :, a:b] * m).sum((1, 2)) / 4
            feats.append(np.bincount(R, s, nr) / np.bincount(R, minlength=nr))
    X.append(np.stack(feats, 1)); y.append(z['pool']); src.append(np.full(nr, len(src)))
X, y, src = np.concatenate(X), np.concatenate(y), np.concatenate(src)
A = np.column_stack([np.ones(len(y)), X])
for name, Aa in [('linear', A), ('linear+sq', np.column_stack([A, X ** 2]))]:
    b, *_ = np.linalg.lstsq(Aa, y, rcond=None)
    res = y - Aa @ b; print(name, 'R2', 1 - res.var() / y.var(), 'resid sd', res.std())
    if name == 'linear':
        k = 1
        print('rounds\\c ' + ' '.join(f'{lo}-{hi}'.rjust(8) for lo, hi in CB))
        for (a2, b2) in TB:
            print(f'{a2}-{b2}'.ljust(8), ' '.join(f'{b[k + j]:8.2f}' for j in range(len(CB)))); k += len(CB)
        print('mean spend per cell (top 50 rules)')
        top = np.argsort(-y)[:50]; ms = X[top].mean(0); k = 0
        for (a2, b2) in TB:
            print(f'{a2}-{b2}'.ljust(8), ' '.join(f'{ms[k + j]:8.2f}' for j in range(len(CB)))); k += len(CB)
        ms = X.mean(0); k = 0; print('mean spend all rules')
        for (a2, b2) in TB:
            print(f'{a2}-{b2}'.ljust(8), ' '.join(f'{ms[k + j]:8.2f}' for j in range(len(CB)))); k += len(CB)
