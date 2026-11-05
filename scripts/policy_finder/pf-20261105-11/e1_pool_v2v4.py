"""v5: pool the v2, v3, v4 real sweeps (v2/v4 = v3 with N = infinity); quadratic surface, profiles, top points."""
import json, numpy as np
B = 'plots/simulation/policy_finder/'
rows = []
for k, v in enumerate(['pf-20261105-11-v2', 'pf-20261105-11-v3', 'pf-20261105-11-v4']):
    for p in json.load(open(B + v + '_sweep/sweep.json'))['points']:
        q = p['params']; rows.append([q['T'], q['A'], np.log(q['B']), q['S'], q.get('N', 30), p['pool'], k])
X = np.array(rows)
good = (X[:, 0] > 11) & (X[:, 0] < 17.5) & (np.exp(X[:, 2]) > 0.9) & (np.exp(X[:, 2]) < 3.5) & (X[:, 1] < 5) & (X[:, 3] >= 5) & (X[:, 4] >= 5)
X = X[good]; print('points', len(X), 'by version', np.bincount(X[:, 6].astype(int)))
T, A, lB, S, N, y, ver = X.T
def feats(T, A, lB, S):
    return np.column_stack([np.ones_like(T), T, T**2, lB, lB**2, T*lB, A, A**2, S, S**2, A*lB, A*T, S*T, S*lB, S*A])
F = feats(T, A, lB, S)
w, *_ = np.linalg.lstsq(F, y, rcond=None)
res = y - F @ w; print('resid sd', res.std().round(2))
g = np.array(np.meshgrid(np.arange(11, 17.01, .25), np.arange(0, 5.01, .25), np.log(np.exp(np.linspace(0, np.log(3.2), 30))), np.arange(5, 17), indexing='ij')).reshape(4, -1)
val = feats(*g) @ w; i = val.argmax(); bt = g[:, i]
print('quad peak', round(val[i], 2), 'T A B S', bt[0], bt[1], np.exp(bt[2]).round(2), bt[3])
def prof(j, vals):
    out = []
    for vv in vals:
        z = bt.copy(); z[j] = vv; out.append(round(float(feats(*[np.array([x]) for x in z]) @ w), 1))
    print('TABS'[j], list(zip(np.round(vals, 2), out)))
prof(0, [11, 12, 13, 14, 15, 16]); prof(1, [0, 1, 2, 3, 4, 5]); prof(2, np.log([1, 1.3, 1.6, 2, 2.5, 3.2])); prof(3, [5, 6, 7, 8, 9, 10, 12, 14, 16])
# within 3 of peak: how many points and what is their mean score / mean residual by version
pred = F @ w
for k in range(3):
    m = (ver == k) & (pred > val[i] - 3)
    print('ver', k + 2, 'near-peak n', m.sum(), 'mean y', y[m].mean().round(2), 'mean pred', pred[m].mean().round(2))
# top 15 raw points
o = np.argsort(-y)[:15]
for j in o: print(np.round([T[j], A[j], np.exp(lB[j]), S[j], N[j], y[j], pred[j], ver[j] + 2], 2))
