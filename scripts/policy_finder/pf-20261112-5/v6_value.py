"""v6: the lead value of a shock by target class. Aligned pairs identical through round R's contributions whose
round-R punishments differ in exactly one member of my group. Regress d lead (per round, rounds R..23; also own
and other pools) on (dose/10) x class dummies, controlling for d (my points dealt after R)/10 and d (my points dealt
in R+1..R+3)/10. Class: c band at R (0-2, 3-5, 6-9, 10+) x newcomer (R>0). Usage: v6_value.py R [versions comma-separated]"""
import numpy as np, itertools, sys
R = int(sys.argv[1]) if len(sys.argv) > 1 else 4
V = sys.argv[2].split(',') if len(sys.argv) > 2 else ['v1', 'v2', 'v3', 'v4', 'v5']
A = {v: np.load(f'/tmp/pf/al_{v}.npz') for v in V}
bands = [(0, 2), (3, 5), (6, 9), (10, 20)]
rows = []
for a, b in itertools.combinations(V, 2):
    za, zb = A[a], A[b]
    same = (za['c'][:, :, :R + 1] == zb['c'][:, :, :R + 1]).all((1, 2)) & (za['p'][:, :, :R] == zb['p'][:, :, :R]).all((1, 2))
    m = za['mine'][:, :, R].astype(bool)
    pa, pb = (za['p'][:, :, R] * m).astype(float), (zb['p'][:, :, R] * m).astype(float)
    nd = (pa != pb).sum(1)
    s = same & (nd == 1)
    if s.sum() == 0: continue
    g = np.nonzero(s)[0]; i = np.argmax((pa != pb)[g], 1)
    dose = (pa - pb)[g, i] / 10
    c = za['c'][g, i, R]
    new = ~za['mine'][g, i, R - 1].astype(bool) if R > 0 else np.zeros(len(g), bool)
    def pools(z):
        cc = z['c'][g][:, :, R:].astype(float); p = z['p'][g][:, :, R:].astype(float); mm = z['mine'][g][:, :, R:].astype(bool)
        pool = 1.6 * cc * z['v'][g][:, :, R:] - p
        return (pool * mm).sum(1).mean(1), (pool * ~mm).sum(1).mean(1)
    oa, xa = pools(za); ob, xb = pools(zb)
    later = lambda z, e: (z['p'][g][:, :, R + 1:e] * z['mine'][g][:, :, R + 1:e]).sum((1, 2)).astype(float) / 10
    rows.append(np.column_stack([dose, c, new, oa - ob, xa - xb, later(za, 24) - later(zb, 24), later(za, R + 4) - later(zb, R + 4)]))
D = np.concatenate(rows)
dose, c, new, down, doth, dlat, dlat3 = D.T
print('R', R, 'pairs', len(D))
names, X = [], []
for lo, hi in bands:
    for nv in ([0, 1] if R > 0 else [0]):
        k = (c >= lo) & (c <= hi) & (new == nv)
        names.append(f'c{lo}-{hi}' + (':new' if nv else (':stay' if R > 0 else ''))); X.append(dose * k)
X = np.column_stack(X + [dlat, dlat3, np.ones(len(D))]); names += ['later/10', 'later3/10', '1']
for lab, y in [('lead', down - doth), ('own', down), ('other', doth)]:
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; s2 = r @ r / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    print(lab, ' '.join(f'{n}:{bb:+.2f}({ss:.2f})' for n, bb, ss in zip(names, b, se)))
for j, n in enumerate(names[:-3]):
    print(n, 'pairs', int((X[:, j] != 0).sum()))
