"""v7: search v6's form (pmax 32) on the pooled, slot-corrected GP of v7_pool.py."""
import sys, pickle, numpy as np
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v7_pool_feats import curve, CS
gp, lo, hi, names = pickle.load(open('/tmp/v7_gp.pkl', 'rb'))
rng = np.random.default_rng(1)
n = 60000
d10 = rng.uniform(4, 9, n); d16 = rng.uniform(0.3, 4, n)
hz = rng.integers(4, 10, n); lt = rng.integers(1, 7, n); ls = rng.integers(11, 19, n)
F = np.array([list(curve('v6', dict(pmax=32, d10=a, d16=min(b, a - 0.01)), CS)) for a, b in zip(d10, d16)])
F = np.column_stack([F, hz, lt, ls])
m, s = gp.predict((F - lo) / (hi - lo), return_std=True)
o = np.argsort(-m)[:12]
for i in o:
    print(f'd10 {d10[i]:.2f} d16 {d16[i]:.2f} h {hz[i]} lthr {lt[i]} lstop {ls[i]}  {m[i]:.2f} +- {s[i]:.2f}')
best = o[0]
def marg(x, nm, edges):
    for a, b in zip(edges[:-1], edges[1:]):
        k = (x >= a) & (x < b)
        print(f'  {nm} {a}-{b}: top-mean {np.sort(m[k])[-200:].mean():.2f}')
marg(d10, 'd10', [4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9])
marg(d16, 'd16', [0.3, 1, 1.5, 2, 2.5, 3, 3.5, 4])
marg(hz, 'horizon', [4, 5, 6, 7, 8, 9, 10])
marg(lt, 'lthr', [1, 2, 3, 4, 5, 6, 7])
marg(ls, 'lstop', [11, 12, 13, 14, 15, 16, 17, 18, 19])
