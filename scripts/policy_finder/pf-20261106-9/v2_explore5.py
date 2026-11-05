# Per design point and opponent: our punishment per member-round by phase vs our pool, members and
# late contributions. Which amount of punishment, when, went with the best pool?
import numpy as np, pandas as pd
d = np.load('/tmp/v1sweep.npz')
c, p, v, g, ours, opp, pt = d['c'].astype(np.float32), d['p'].astype(np.float32), d['v'], d['g'], d['ours'], d['opp'], d['pt']
ing = (g == ours[:, None, None])
n = ing.sum(1)                                     # [E,24]
pool = ((1.6 * c - p) * ing).sum(1)                 # [E,24]
ph = (p * ing).sum(1)
cm = (c * ing).sum(1) / np.maximum(n, 1)
df = pd.DataFrame(dict(pt=pt, opp=opp, pool=pool.mean(1), mem=n.mean(1),
    p0=ph[:, 0:2].sum(1) / np.maximum(n[:, 0:2].sum(1), 1),
    p1=ph[:, 2:8].sum(1) / np.maximum(n[:, 2:8].sum(1), 1),
    p2=ph[:, 8:16].sum(1) / np.maximum(n[:, 8:16].sum(1), 1),
    p3=ph[:, 16:].sum(1) / np.maximum(n[:, 16:].sum(1), 1),
    ptot=ph.sum(1) / np.maximum(n.sum(1), 1),
    clate=cm[:, 16:].mean(1), pool_late=pool[:, 16:].mean(1)))
A = df.groupby(['pt', 'opp']).mean().reset_index()
for col in ['p0', 'ptot']:
    A['b'] = pd.qcut(A[col], 8)
    print(f'by {col} (per design point x opponent)')
    print(A.groupby(['b'], observed=True)[['pool', 'mem', 'clate', 'pool_late', 'p0', 'p1', 'p2', 'p3']].mean().round(2).to_string())
    print(A.pivot_table(index='b', columns='opp', values='pool', observed=True).round(1).to_string())
X = np.column_stack([np.ones(len(A)), A.p0, A.p1, A.p2, A.p3, A.p0**2, A.p1**2])
for o in [1, 2, 3, 4]:
    m = A.opp == o
    b = np.linalg.lstsq(X[m], A.pool[m], rcond=None)[0]
    print('opp', o, 'pool ~ 1 p0 p1 p2 p3 p0^2 p1^2:', np.round(b, 2))
b = np.linalg.lstsq(X, A.pool, rcond=None)[0]
print('all', np.round(b, 2))
