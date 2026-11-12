"""v7: value of a round-R shock by how many members it hits / how many low contributors the group has
(/tmp/pf/pairs.npz). d lead over R..23 per 10 points, controls for later points (3 phases)."""
import numpy as np, sys
z = np.load('/tmp/pf/pairs.npz')
for R0 in [0, 4, 8]:
    i = (z['R'] == R0) & (z['dd'] != 0)
    s = np.sign(z['dd'][i]).astype(float)
    dd = z['dd'][i] * s / 10; late = z['late'][i] * s[:, None] / 10
    nhit = np.where(s > 0, z['na'][i], z['nb'][i]); nlow = z['nlow'][i]; n = z['n'][i]; frac = nlow / np.maximum(n, 1)
    def fit(bins, label, ys=('dlead', 'dmem')):
        X = np.stack([dd * b for b in bins.values()] + [late[:, j] for j in range(3)] + [b * 1.0 for b in bins.values()], 1)
        print(f'R{R0} -- {label}')
        for k in ys:
            y = z[k][i] * s
            b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
            s2 = (r ** 2).sum() / (len(y) - X.shape[1]); se = np.sqrt(np.abs(np.diag(s2 * np.linalg.pinv(X.T @ X))))
            print(f'  {k:6s}', ' '.join(f'{nm}:{b[j]:+.2f}({se[j]:.2f})' for j, nm in enumerate(bins)))
        print('  n', {nm: int(bb.sum()) for nm, bb in bins.items()})
    fit({f'hit{a}': (nhit == a) if a < 4 else (nhit >= 4) for a in [1, 2, 3, 4]}, 'members hit in the higher-dose game', ('dlead', 'down', 'doth', 'dmem'))
    fit({f'low{a}': (nlow == a) if a < 4 else (nlow >= 4) for a in [1, 2, 3, 4]}, 'members giving <=5')
    fit({'f<=.25': frac <= .25, 'f.25-.5': (frac > .25) & (frac <= .5), 'f>.5': frac > .5}, 'share of my group giving <=5')
