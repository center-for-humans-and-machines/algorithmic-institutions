# v6: exits from my group at each decision, per opponent, against the player's own punishment
# in the block (by round within block) and the other group's punishment level. v5 sweep.
import numpy as np
d = np.load('/tmp/v5sweep.npz')
C, P, G, V = [d[k].astype(np.float32) for k in 'CPGV']; opp = d['opp']; seat = d['seat']
mine = G == seat[:, None, None]
for dec in [3, 7, 11]:
    b0 = dec - 3
    m = mine[:, :, b0:dec + 1].all(2)  # in my group the whole block
    left = (G[:, :, dec + 1] != G[:, :, dec])
    oth = (~mine[:, :, b0]).astype(np.float32)
    oppp = (P[:, :, b0:dec + 1] * oth[..., None]).sum((1, 2)) / np.maximum(oth.sum(1), 1) / 4
    for o in [1, 2, 3, 4]:
        s = m & (opp[:, None] == o)
        y = left[s].astype(float)
        X = [np.ones(s.sum())] + [np.sqrt(P[:, :, r][s]) for r in range(b0, dec)] + [C[:, :, b0:dec + 1].mean(2)[s]]
        X = np.stack(X, 1)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        unp = s & (P[:, :, b0:dec + 1].sum(2) == 0)
        print(f'dec {dec} opp{o} n={s.sum():7d} exit {y.mean():.3f} unpunished {left[unp].mean():.3f}  other ppm {oppp[opp==o].mean():.2f}  per sqrt p by round', np.round(b[1:4], 4), 'per c', round(b[4], 4))
