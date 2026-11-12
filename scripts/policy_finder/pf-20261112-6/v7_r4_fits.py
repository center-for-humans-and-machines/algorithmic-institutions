"""v7: fits on the pooled clean round-4 experiment with v6 added (v7_r4_pool4.py). Same model as v6_r4_fits.py
(dose and dose^2/10 per class, residualized on their design expectation; source dummies, own size, seat)."""
import numpy as np, sys
D = dict(np.load('/tmp/v7_pool4.npz'))
here = D['here'].astype(bool); c4 = D['c4']; act = D['act']; src = D['src']
Y = {k: D[k] for k in ['lead', 's8', 's12', 'ownc']}

def fit(classes, sel=None, title='', doses=(8, 13, 18, 23, 28, 30), ys=('lead', 's8', 'ownc'), lin=False):
    s = np.ones(len(c4), bool) if sel is None else sel
    R, E = [], []
    for lab, cm in classes:
        cm = cm & here
        a1 = (act * cm).sum(1); e1 = (D['e1'] * cm).sum(1)
        a2 = (act ** 2 / 10 * cm).sum(1); e2 = (D['e2'] * cm).sum(1)
        R += [a1 - e1] + ([] if lin else [a2 - e2]); E += [e1] + ([] if lin else [e2])
    X = np.column_stack([np.ones(len(c4))] + R + E + [src == 1, src == 2, src == 3, D['size4'], D['seat']])[s].astype(float)
    X = X[:, X.std(0) > 0 if X.shape[1] else slice(None)] if False else X
    print('==', title, 'games', s.sum())
    k = 1 if lin else 2
    for yk in ys:
        y = Y[yk][s]; bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb
        V = np.linalg.pinv(X.T @ X) * (e ** 2).mean()
        for i, (lab, _) in enumerate(classes):
            line = f'  {yk:5s} {lab:16s}'
            for dd in doses:
                gv = np.zeros(len(bb)); gv[1 + k * i] = dd
                if not lin: gv[2 + k * i] = dd ** 2 / 10
                line += f' d{dd} {gv @ bb:+.2f}({np.sqrt(gv @ V @ gv):.2f})'
            if not lin:
                gv = np.zeros(len(bb)); gv[1 + k * i] = 1; gv[2 + k * i] = 2 * 28 / 10
                line += f' | slope@28 {gv @ bb:+.3f}({np.sqrt(gv @ V @ gv):.3f})'
            print(line)

rng = lambda a, z: (c4 >= a) & (c4 <= z)
G = [(f'c{a}-{z}', rng(a, z)) for a, z in [(0, 0), (1, 2), (3, 5), (6, 7), (8, 10)]]
fit(G, title='pooled v3-v6, grouped', ys=('lead', 's8', 's12', 'ownc'))
fit(G, sel=np.isin(src, [0, 3]), title='v5+v6 only')
fit([(f'c{a}-{z}', rng(a, z)) for a, z in [(0, 0), (1, 2), (3, 5)]], sel=src == 3, title='v6 only, linear', lin=True,
    doses=(1, 14, 21, 28))
sz = D['size4']
for a, z in [(1, 4), (5, 6), (7, 8)]:
    fit(G, sel=(sz >= a) & (sz <= z), title=f'own size {a}-{z}', ys=('lead',))
