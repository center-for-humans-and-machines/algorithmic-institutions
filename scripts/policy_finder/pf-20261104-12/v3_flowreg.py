"""v3: OLS of the rule's group's net member gain at each decision (joiners - leavers) on
the two groups' block mean contributions, the rule group's punishment per member in each
round of the block, the share of punished members, the sizes. Per decision."""
import sys, numpy as np, pandas as pd
rows = []
for f in sys.argv[1:]:
    z = np.load(f); c, p, g = (z[k].astype(np.int16) for k in 'cpg')
    for dec in (3, 7, 11, 15, 19):
        b = slice(dec-3, dec+1)
        ours = g[..., dec] == 0; theirs = g[..., dec] == 1
        n0 = ours.sum(-1); n1 = theirs.sum(-1)
        gm = (g[..., b] == 0); hm = (g[..., b] == 1)
        D = {'dec': dec, 'n0': n0, 'n1': n1}
        D['c0'] = (c[..., b]*gm).sum((-1, -2)) / np.maximum(gm.sum((-1, -2)), 1)
        D['c1'] = (c[..., b]*hm).sum((-1, -2)) / np.maximum(hm.sum((-1, -2)), 1)
        D['c0last'] = (c[..., dec]*ours).sum(-1) / np.maximum(n0, 1)
        D['c1last'] = (c[..., dec]*theirs).sum(-1) / np.maximum(n1, 1)
        for k in range(4):
            t = dec-3+k; m = g[..., t] == 0
            D[f'p0_{k}'] = (p[..., t]*m).sum(-1) / np.maximum(m.sum(-1), 1)
            D[f'max0_{k}'] = (p[..., t]*m).max(-1)
        D['p1'] = (p[..., b]*hm).sum((-1, -2)) / np.maximum(hm.sum((-1, -2)), 1)
        D['net'] = (theirs & (g[..., dec+1] == 0)).sum(-1) - (ours & (g[..., dec+1] != 0)).sum(-1)
        rows.append(pd.DataFrame({k: (np.broadcast_to(v, n0.shape).ravel()) for k, v in D.items()}))
d = pd.concat(rows); d = d[(d.n0 > 0) & (d.n1 > 0)]
X0 = ['c0', 'c1', 'c0last', 'c1last', 'p0_0', 'p0_1', 'p1', 'n0']
for dec, x in d.groupby('dec'):
    X = x[X0].astype(float).copy(); X['const'] = 1
    beta = np.linalg.lstsq(X.to_numpy(), x.net.to_numpy().astype(float), rcond=None)[0]
    print(dec, len(x), dict(zip(X.columns, beta.round(3))))
x = d[d.dec == 3]
X = x[['c0', 'c1', 'c0last', 'c1last', 'p0_0', 'p0_1', 'max0_0', 'max0_1', 'p1', 'n0']].astype(float).copy(); X['const'] = 1
print('with max', dict(zip(X.columns, np.linalg.lstsq(X.to_numpy(), x.net.to_numpy().astype(float), rcond=None)[0].round(3))))
