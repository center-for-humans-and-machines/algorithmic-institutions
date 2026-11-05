# v7: at later block starts (4, 8, 12), arrivals vs stayers in my group: lift and exit per sqrt point of
# the block-start punishment, within cells of own c at b0, opponent, group size and received bin.
# v6 sweep (random params give the variation). Also the share of arrivals never punished.
import numpy as np, sys
def fe(y, x, cell):
    _, inv = np.unique(cell, return_inverse=True); cnt = np.bincount(inv)
    def dm(v): return v - (np.bincount(inv, v) / cnt)[inv]
    X = np.stack([dm(c) for c in x], 1); Y = dm(y)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b; s2 = (r**2).sum() / max(len(Y) - X.shape[1] - len(cnt), 1)
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
d = np.load(sys.argv[1] if len(sys.argv) > 1 else '/tmp/v6sweep.npz')
C, P, G, V = [d[k].astype(np.float32) for k in 'CPGV']; opp = d['opp']; seat = d['seat']
mine = G == seat[:, None, None]
for b0 in [4, 8, 12]:
    dec = b0 + 3
    m = mine[:, :, b0:dec + 1].all(2) & (V[:, :, b0] > 0)
    arr = ~mine[:, :, b0 - 1]
    n = mine[:, :, b0].sum(1, keepdims=True)
    rec = P[:, :, :b0].sum(2)
    recb = np.digitize(rec, [0.5, 10, 25, 50])
    left = (G[:, :, dec + 1] != G[:, :, dec]).astype(float)
    cell = (((C[:, :, b0] * 5 + opp[:, None]) * 9 + n) * 5 + recb) * 2 + arr
    sp = np.sqrt(P[:, :, b0])
    for nm, g in [('stayers', ~arr), ('arrivals', arr)]:
        for rn, rg in [('never punished', rec == 0), ('punished before', rec > 0)]:
            mm = m & g & rg
            b, se = fe(left[mm], [sp[mm]], cell[mm])
            st = mm & (G[:, :, dec + 4] == G[:, :, dec]) & (G[:, :, dec + 1] == G[:, :, dec])
            y = C[:, :, dec + 4] - C[:, :, b0]
            b2, se2 = fe(y[st], [sp[st]], cell[st])
            print(f'b0 {b0} {nm:8s} {rn:15s} n/game {mm.sum() / len(C):.2f} c {C[:, :, b0][mm].mean():.1f} p {P[:, :, b0][mm].mean():.1f}'
                  f' | exit/sqrt {b[0]:+.4f}({se[0]:.4f}) | lift c(b0+7) /sqrt {b2[0]:+.3f}({se2[0]:.3f})')
    for o in np.unique(opp):
        so = (opp == o)[:, None] & m & arr
        print(f'    opp {o}: arrivals/game {so.sum() / (opp == o).sum():.2f}, never punished share {(rec[so] == 0).mean():.2f}, c {C[:, :, b0][so].mean():.1f}')
