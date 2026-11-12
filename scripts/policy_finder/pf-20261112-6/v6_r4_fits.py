"""v6: fits on the pooled clean round-4 experiment (v6_r4_pool3.py). Total effect of hitting one member of a class
with d points (dose and dose^2/10 terms, residualized on their design expectation), source dummies and own size as
controls."""
import numpy as np
D = dict(np.load('/tmp/v6_pool3.npz'))
here = D['here'].astype(bool); c4 = D['c4']; act = D['act']
Y = {k: D[k] for k in ['lead', 's8', 's12', 'ownc']}
src = D['src']

def fit(classes, sel=None, title='', doses=(8, 13, 18, 23, 28), ys=('lead', 's8', 'ownc'), quiet=False):
    s = np.ones(len(c4), bool) if sel is None else sel
    R, E = [], []
    for lab, cm in classes:
        cm = cm & here
        a1 = (act * cm).sum(1); e1 = (D['e1'] * cm).sum(1)
        a2 = (act ** 2 / 10 * cm).sum(1); e2 = (D['e2'] * cm).sum(1)
        R += [a1 - e1, a2 - e2]; E += [e1, e2]
    X = np.column_stack([np.ones(len(c4))] + R + E + [src == 1, src == 2, D['size4'], D['seat']])[s].astype(float)
    res = {}
    if not quiet: print('==', title, 'games', s.sum())
    for k in ys:
        y = Y[k][s]; bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb
        V = np.linalg.pinv(X.T @ X) * (e ** 2).mean()
        res[k] = (bb, V)
        for i, (lab, _) in enumerate(classes):
            line = f'  {k:5s} {lab:20s}'
            for dd in doses:
                gv = np.zeros(len(bb)); gv[1 + 2 * i] = dd; gv[2 + 2 * i] = dd ** 2 / 10
                line += f' d{dd} {gv @ bb:+.2f}({np.sqrt(gv @ V @ gv):.2f})'
            if not quiet: print(line)
    return res

if __name__ == "__main__" and "--new" not in __import__("sys").argv:
    print('games per source', np.bincount(src.astype(int)))
    rng = lambda a, z: (c4 >= a) & (c4 <= z)
    fit([(f'c{a}-{z}', rng(a, z)) for a, z in [(0, 0), (1, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10)]], title='by value')
    G = [(f'c{a}-{z}', rng(a, z)) for a, z in [(0, 2), (3, 5), (6, 7), (8, 10)]]
    fit(G, title='grouped', ys=('lead', 's8', 's12', 'ownc'))
    sz = D['size4']
    for a, z in [(1, 4), (5, 6), (7, 8)]:
        fit(G, sel=(sz >= a) & (sz <= z), title=f'own size {a}-{z}')
    oth = D['oth'][:, None] if D['oth'].ndim == 1 else D['oth']
    gap = oth - c4
    fit([('c0-5 gap<6', rng(0, 5) & (gap < 6)), ('c0-5 gap>=6', rng(0, 5) & (gap >= 6)),
         ('c6-7 gap<6', rng(6, 7) & (gap < 6)), ('c6-7 gap>=6', rng(6, 7) & (gap >= 6)),
         ('c8-10', rng(8, 10))], title='by mates mean minus own c4')
    fit([('c0-5 drop', rng(0, 5) & (D['c3'] - c4 >= 3)), ('c0-5 nodrop', rng(0, 5) & (D['c3'] - c4 < 3)),
         ('c6-7', rng(6, 7))], title='by drop from round 3')

if "--new" in __import__("sys").argv:
    rng = lambda a, z: (c4 >= a) & (c4 <= z)
    new = D['new4'].astype(bool)
    print('newcomers among 0-5 givers at round 4:', round(float((new & rng(0, 5) & here).sum() / (rng(0, 5) & here).sum()), 3))
    fit([('c0-2 stayer', rng(0, 2) & ~new), ('c0-2 newcomer', rng(0, 2) & new), ('c3-5 stayer', rng(3, 5) & ~new),
         ('c3-5 newcomer', rng(3, 5) & new), ('c6-7', rng(6, 7))], title='newcomers (joined at the round-3 decision)',
        ys=('lead', 's8', 's12', 'ownc'))
