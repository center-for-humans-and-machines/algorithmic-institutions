"""Point-level: lead (pool difference per round) of every v1 and v2 sweep point against the punishment it dealt per
member-round in each phase (rounds 0-3, 4-7, 8-19, 20-23). OLS with linear and squared terms; the points' params set
those amounts, so this is a rough across-point read of which phase's punishment costs or pays."""
import numpy as np, pandas as pd, numpy.linalg as la
rows = []
for f in ['/tmp/pf/v1.npz', '/tmp/pf/v2.npz']:
    z = np.load(f); c, p, m, v = [z[k].astype(np.float32) for k in ['c', 'p', 'mine', 'v']]
    pt = z['point'].astype(int)
    lead = ((1.6 * c * v - p) * m - (1.6 * c * v - p) * (1 - m)).sum(1).mean(1)
    d = dict(pt=pt, lead=lead, nm=(m * v).sum(1).mean(1))
    for a, b in [(0, 4), (4, 8), (8, 20), (20, 24)]:
        d[f'p{a}'] = (p * m)[:, :, a:b].sum((1, 2)); d[f'n{a}'] = (m * v)[:, :, a:b].sum((1, 2))
    g = pd.DataFrame(d).groupby('pt').mean()
    for a in [0, 4, 8, 20]: g[f'q{a}'] = g[f'p{a}'] / g[f'n{a}']
    g['src'] = f[-6:-4]; rows.append(g)
G = pd.concat(rows)
X = G[['q0', 'q4', 'q8', 'q20']].copy()
for a in ['q0', 'q4', 'q8']: X[a + '^2'] = X[a] ** 2
X['v2'] = (G.src == 'v2').astype(float)
Xm = np.column_stack([np.ones(len(X)), X.values]); y = G.lead.values
b, *_ = la.lstsq(Xm, y, rcond=None); r = y - Xm @ b; se = np.sqrt(np.diag(r.var() * la.inv(Xm.T @ Xm)))
print('lead ~', ' '.join(f'{n}={b[i+1]:+.2f}({se[i+1]:.2f})' for i, n in enumerate(X.columns)), f'const={b[0]:.2f} resid sd={r.std():.2f}')
for col in ['q0', 'q4', 'q8']:
    G['b'] = pd.cut(G[col], [-0.01, 0.1, 0.5, 1, 2, 4, 30])
    print(col, G.groupby('b', observed=True).agg(lead=('lead', 'mean'), n=('lead', 'size'), q8=('q8', 'mean'), nm=('nm', 'mean')).round(2).T.to_string())
