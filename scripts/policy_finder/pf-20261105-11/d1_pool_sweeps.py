"""v4: pool the v2 and v3 real sweeps (v2 = v3 with N = infinity) and fit a quadratic surface."""
import json, numpy as np
B = 'plots/simulation/policy_finder/'
rows = []
for v, hasN in [('pf-20261105-11-v2', 0), ('pf-20261105-11-v3', 1)]:
    for p in json.load(open(B + v + '_sweep/sweep.json'))['points']:
        q = p['params']; rows.append([q['T'], q['A'], np.log(q['B']), q['S'], q.get('N', 30), p['pool'], hasN])
X = np.array(rows)
good = (X[:, 0] > 11) & (X[:, 0] < 17.5) & (np.exp(X[:, 2]) > 0.9) & (np.exp(X[:, 2]) < 3.5) & (X[:, 1] < 4.5) & (X[:, 3] >= 5)
X = X[good]; print('points in common region', len(X), 'v3', X[:, 6].sum())
T, A, lB, S, N, y, v3 = X.T
Nc = np.minimum(N, 14)
F = np.column_stack([np.ones_like(T), T, T**2, lB, lB**2, T*lB, A, A**2, S, S**2, Nc, Nc**2, S*Nc, A*lB, A*T])
names = '1 T T2 lB lB2 TlB A A2 S S2 N N2 SN AlB AT'.split()
w, *_ = np.linalg.lstsq(F, y, rcond=None)
res = y - F @ w; print('resid sd', res.std().round(2))
for n, x in zip(names, w): print(f'{n:4s} {x:9.3f}')
# grid optimum
best = None
for t in np.arange(11, 17.01, .25):
  for b in np.exp(np.linspace(0, np.log(3.2), 30)):
    for a in np.arange(0, 4.01, .25):
      for s in range(5, 17):
        for n in list(range(3, 15)):
          f = np.array([1, t, t*t, np.log(b), np.log(b)**2, t*np.log(b), a, a*a, s, s*s, n, n*n, s*n, a*np.log(b), a*t])
          val = f @ w
          if best is None or val > best[0]: best = (val, t, a, b, s, n)
print('quad peak', np.round(best, 2))
def prof(idx, vals):
    out = []
    for vv in vals:
        t, a, b, s, n = best[1:]
        z = dict(t=t, a=a, b=b, s=s, n=n); z[idx] = vv
        t, a, b, s, n = z['t'], z['a'], z['b'], z['s'], z['n']
        f = np.array([1, t, t*t, np.log(b), np.log(b)**2, t*np.log(b), a, a*a, s, s*s, n, n*n, s*n, a*np.log(b), a*t])
        out.append(round(float(f @ w), 1))
    print(idx, list(zip(vals, out)))
prof('s', list(range(5, 17, 2))); prof('n', [3, 4, 5, 6, 8, 10, 14]); prof('t', [11, 12, 13, 14, 15, 16, 17]); prof('a', [0, 1, 2, 3, 4]); prof('b', [1, 1.3, 1.6, 2, 2.5, 3.2])
# mean score by v and top points by version
for k in [0, 1]:
    m = v3 == k; print('v3' if k else 'v2', 'n', m.sum(), 'mean', y[m].mean().round(1), 'top10 mean', np.sort(y[m])[-10:].mean().round(1))
