"""v5: pool v2-v4 sweep points into v4's parametrisation and fit one GP over all of them."""
import json, numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
B = 'plots/simulation/policy_finder/'
rows = []
def add(v, f):
    for p in json.load(open(B + v + '_sweep/sweep.json'))['points']:
        rows.append(f(p['params']) + [p['pool'], int(v[-1]) if v[-2] == 'v' else 1])
add('pf-20261105-8-v2', lambda q: [q['standard'], q['standard'], q['curvature'] / q['standard'], q['dose'], q['horizon'], 1.0, q['horizon']])
add('pf-20261105-8-v3', lambda q: [q['standard'], q['standard'], q['curvature'] / q['standard'], q['dose'], q['horizon'], q['floor'], q['horizon'] + q['tail']])
add('pf-20261105-8-v4', lambda q: [q['opening'], q['standard'], q['rate'], q['dose'], q['horizon'], q['floor'], max(q['end'], q['horizon'])])
A = np.array(rows); names = ['open', 'std', 'rate', 'dose', 'hor', 'floor', 'end']
X = A[:, :7].copy(); y = A[:, 7]; ver = A[:, 8]
X[:, 2] = np.log(X[:, 2]); X[:, 5] = np.log(X[:, 5])
X[:, 3] = np.minimum(X[:, 3], 45)
X[:, 6] = X[:, 6] - X[:, 4]   # tail length
names[6] = 'tail'
lo, hi = X.min(0), X.max(0); Z = (X - lo) / (hi - lo)
keep = y > 600
gp = GaussianProcessRegressor(C(100) * RBF(np.ones(7) * .3, (1e-2, 1e2)) + WhiteKernel(10), normalize_y=True, n_restarts_optimizer=3, random_state=0)
gp.fit(Z[keep], y[keep])
print(gp.kernel_)
mu = gp.predict(Z)
print('resid sd', np.std(y[keep] - mu[keep]))
order = np.argsort(-mu)
for i in order[:20]:
    print(int(ver[i]), np.round(A[i, :7], 3), 'true', round(y[i], 1), 'fit', round(mu[i], 1))
# random search on the fitted surface within data hull
rng = np.random.default_rng(0)
Q = rng.random((20000, 7)); m, s = gp.predict(Q, return_std=True)
j = np.argsort(-(m - s))[:10]
for k in j:
    x = lo + Q[k] * (hi - lo); x[2] = np.exp(x[2]); x[5] = np.exp(x[5])
    print('cand', np.round(x, 3), round(m[k], 1), round(s[k], 1))
np.save('/tmp/x/pool.npy', A)

best = Q[j[0]].copy()
for d in range(7):
    g = np.linspace(0, 1, 9); M = np.repeat(best[None], 9, 0); M[:, d] = g
    v = lo[d] + g * (hi[d] - lo[d])
    if d in (2, 5): v = np.exp(v)
    print(names[d], ' '.join(f'{a:.3g}:{b:.1f}' for a, b in zip(v, gp.predict(M))))
