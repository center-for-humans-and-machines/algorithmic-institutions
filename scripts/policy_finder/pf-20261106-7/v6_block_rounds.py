# v6: within cells of the player's own contributions in rounds b0..b0+2 (exact), the opponent and
# the group size, the exit at the block's decision and the later contribution of stayers, per
# sqrt point of punishment in each of the block's first three rounds. v5 sweep (random params).
import numpy as np, sys
def fe(y, x, cell):
    _, inv = np.unique(cell, return_inverse=True); cnt = np.bincount(inv)
    def dm(v): return v - (np.bincount(inv, v) / cnt)[inv]
    X = np.stack([dm(c) for c in x], 1); Y = dm(y)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b; s2 = (r**2).sum() / max(len(Y) - X.shape[1] - len(cnt), 1)
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
d = np.load(sys.argv[1] if len(sys.argv) > 1 else '/tmp/v5sweep.npz')
C, P, G, V = [d[k].astype(np.float32) for k in 'CPGV']; opp = d['opp']; seat = d['seat']
mine = G == seat[:, None, None]
for b0 in [0, 4, 8]:
    dec = b0 + 3
    m = mine[:, :, b0:dec + 1].all(2) & (V[:, :, b0:b0 + 3] > 0).all(2)
    n = mine[:, :, b0].sum(1, keepdims=True)
    left = (G[:, :, dec + 1] != G[:, :, dec]).astype(float)
    cell = ((C[:, :, b0] * 21 + C[:, :, b0 + 1]) * 21 + C[:, :, b0 + 2]) * 5 + opp[:, None]
    cell = cell * 9 + n
    if b0 > 0:  # also condition on punishment received before, binned
        rec = np.minimum(np.sqrt(P[:, :, :b0].sum(2)) // 2, 6)
        cell = cell * 7 + rec
    x = [np.sqrt(P[:, :, r])[m] for r in range(b0, b0 + 3)]
    b, se = fe(left[m], x, cell[m])
    st = m & (G[:, :, dec + 5] == G[:, :, dec]) & (G[:, :, dec + 1] == G[:, :, dec])
    y = C[:, :, dec + 5] - C[:, :, b0 + 2]
    xs = [np.sqrt(P[:, :, r])[st] for r in range(b0, b0 + 3)]
    b2, se2 = fe(y[st], xs, cell[st])
    print(f'block start {b0}: n={m.sum()} exit per sqrt p r0/r1/r2', ' '.join(f'{v:+.4f}({s:.4f})' for v, s in zip(b, se)),
          '| stayers dc(dec+5) per sqrt p', ' '.join(f'{v:+.3f}({s:.3f})' for v, s in zip(b2, se2)))
