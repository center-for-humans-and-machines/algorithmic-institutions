"""Finer version of v3_phase.py: punishment per member-round in rounds 0, 1-3, 4, 5-7, 8-11, 12-19, 20-23 (linear
and squared), point-level OLS of the lead on all; and the implied best dose per member-round per phase."""
import numpy as np, pandas as pd, numpy.linalg as la
rows = []
ph = [(0, 1), (1, 4), (4, 5), (5, 8), (8, 12), (12, 20), (20, 24)]
for f in ['/tmp/pf/v1.npz', '/tmp/pf/v2.npz']:
    z = np.load(f); c, p, m, v = [z[k].astype(np.float32) for k in ['c', 'p', 'mine', 'v']]
    pt = z['point'].astype(int)
    lead = ((1.6 * c * v - p) * m - (1.6 * c * v - p) * (1 - m)).sum(1).mean(1)
    d = dict(pt=pt, lead=lead)
    for a, b in ph:
        d[f'p{a}'] = (p * m)[:, :, a:b].sum((1, 2)); d[f'n{a}'] = (m * v)[:, :, a:b].sum((1, 2))
    g = pd.DataFrame(d).groupby('pt').mean()
    for a, b in ph: g[f'q{a}'] = g[f'p{a}'] / g[f'n{a}']
    g['src'] = f[-6:-4]; rows.append(g)
G = pd.concat(rows)
qs = [f'q{a}' for a, b in ph]
X = G[qs].copy()
for a in qs[:-1]: X[a + '^2'] = X[a] ** 2
X['v2'] = (G.src == 'v2').astype(float)
Xm = np.column_stack([np.ones(len(X)), X.values]); y = G.lead.values
b, *_ = la.lstsq(Xm, y, rcond=None); r = y - Xm @ b; se = np.sqrt(np.diag(r.var() * la.inv(Xm.T @ Xm)))
co = dict(zip(X.columns, b[1:])); cse = dict(zip(X.columns, se[1:]))
for q in qs:
    s = f'{q}: lin {co[q]:+.2f}({cse[q]:.2f})'
    if q + '^2' in co:
        s += f' sq {co[q+"^2"]:+.2f}({cse[q+"^2"]:.2f})'
        if co[q + '^2'] < 0 and co[q] > 0:
            x = -co[q] / (2 * co[q + '^2']); s += f'  best {x:.2f}/member-round, gain {co[q]*x + co[q+"^2"]*x*x:+.2f}'
    print(s, f' (range of q in data: 90% < {G[q].quantile(.9):.2f})')
print('corr of phases'); print(G[qs].corr().round(2))
