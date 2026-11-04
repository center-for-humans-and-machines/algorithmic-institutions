"""v2: on v1's sweep, what draws ah's members to our group at a decision, and how the
group's state at the stop round relates to the point's pool."""
import numpy as np, pandas as pd
a = np.load('/tmp/v2_arrays.npz')
C, P, G, V = [a[k].astype(np.float32) for k in 'CPGV']
prm, pool = a['prm'], a['pool']
Pn, E, N, R = C.shape
rows = []
for r in [3, 7, 11, 15, 19]:
    blk = slice(r - 3, r + 1)
    in0 = (G[..., blk] == 0); in1 = ~in0
    n0 = in0.sum(2).mean(-1); n1 = in1.sum(2).mean(-1)          # [P,E]
    c0 = (C[..., blk] * in0).sum((2, 3)) / np.maximum(in0.sum((2, 3)), 1)
    c1 = (C[..., blk] * in1).sum((2, 3)) / np.maximum(in1.sum((2, 3)), 1)
    p0 = (P[..., blk] * in0).sum((2, 3)) / np.maximum(in0.sum((2, 3)), 1)
    p1 = (P[..., blk] * in1).sum((2, 3)) / np.maximum(in1.sum((2, 3)), 1)
    in1_now = G[..., r] == 1
    join = (in1_now & (G[..., r + 1] == 0)).sum(2) / np.maximum(in1_now.sum(2), 1)
    leave_n = ((G[..., r] == 0) & (G[..., r + 1] == 1)).sum(2) / np.maximum((G[..., r] == 0).sum(2), 1)
    rows.append(pd.DataFrame({'r': r, 'n0': n0.ravel(), 'dc': (c0 - c1).ravel(), 'p0': p0.ravel(), 'p1': p1.ravel(),
                              'join': join.ravel(), 'leave': leave_n.ravel(), 'has1': (in1_now.sum(2) > 0).ravel()}))
d = pd.concat(rows); d = d[d.has1]
d['dcb'] = pd.cut(d.dc, [-20, -4, -2, 0, 2, 4, 20]); d['p0b'] = pd.cut(d.p0, [-1, 0.01, 1, 2, 4, 8, 30])
print('join rate of ah members by our-minus-ah mean contribution (block) x our mean punishment')
print(d.groupby(['dcb', 'p0b'], observed=True).join.mean().unstack().round(3).to_string())
print(d.groupby('p0b', observed=True)[['join', 'leave']].mean().round(3).to_string())
print('by round, p0 == 0 vs > 2:')
print(d.assign(pun=d.p0 > 2).groupby(['r', 'pun'])[['join', 'leave', 'dc']].mean().round(3).unstack().to_string())
print('join, rounds 7-19 only, by dc bin x p0 bin')
x = d[d.r >= 7]
print(x.groupby(['dcb', 'p0b'], observed=True).join.agg(['mean']).unstack().round(3).to_string())
print(x.groupby(['dcb', 'p0b'], observed=True).join.size().unstack().to_string())
