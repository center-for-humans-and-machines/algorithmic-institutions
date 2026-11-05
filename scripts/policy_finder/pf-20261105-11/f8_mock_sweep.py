"""v6: mock sweeps. Truth = the pooled v2-v4 quadratic of e6 (interior peak) with S=8, plus an assumed opening-offset
effect -k*(D - D*)^2 (D* = -0.7, as the v3 surrogate opening search hinted; k so that D=-4 costs 3). Noise sd 3.3 per point.
Design A: T, A, log B (D fixed 0). Design B: T, A, log B, D in [-4, 1.5]. Fit quadratic, true value at fitted peak."""
import json, numpy as np
from scipy.stats import qmc
B = 'plots/simulation/policy_finder/'
rows = []
for v in ['pf-20261105-11-v2', 'pf-20261105-11-v3', 'pf-20261105-11-v4']:
    for p in json.load(open(B + v + '_sweep/sweep.json'))['points']:
        q = p['params']; rows.append([q['T'], q['A'], np.log(q['B']), q['S'], q.get('N', 30), p['pool']])
X = np.array(rows)
g = (X[:, 0] > 11) & (X[:, 0] < 17.5) & (np.exp(X[:, 2]) > 0.9) & (np.exp(X[:, 2]) < 3.5) & (X[:, 1] < 5) & (X[:, 3] >= 5) & (X[:, 4] >= 5)
X = X[g]
def feats(T, A, lB, S):
    return np.column_stack([np.ones_like(T), T, T**2, lB, lB**2, T*lB, A, A**2, S, S**2, A*lB, A*T, S*T, S*lB, S*A])
w, *_ = np.linalg.lstsq(feats(*X[:, :4].T), X[:, 5], rcond=None)
Ds, k = -0.7, 3 / 3.3**2
def truth(T, A, lB, D, Dgain=True):
    return feats(T, A, lB, np.full_like(T, 8.0)) @ w - (k * (D - Ds)**2 if Dgain else 0 * D)
def ff(T, A, lB, D, useD):
    F = [np.ones_like(T), T, T**2, lB, lB**2, T*lB, A, A**2, A*lB, A*T]
    if useD: F += [D, D**2, D*T, D*lB, D*A]
    return np.column_stack(F)
gr = np.array(np.meshgrid(np.arange(11, 16.01, .25), np.arange(0, 5.01, .25), np.linspace(np.log(1.1), np.log(3.2), 25), np.arange(-4, 1.51, .25), indexing='ij')).reshape(4, -1)
for scen, dg in [('D matters (peak -0.7)', True), ('D irrelevant', False)]:
    ceil = truth(*gr, Dgain=dg).max()
    for name, (lo, hi, useD) in {'3D (D=0)': ((11.5, 0.5, np.log(1.3), 0), (15, 5, np.log(3.0), 0), False),
                                  '4D D[-4,1.5]': ((11.5, 0.5, np.log(1.3), -4), (15, 5, np.log(3.0), 1.5), True),
                                  '4D D[-3,1]': ((11.5, 0.5, np.log(1.3), -3), (15, 5, np.log(3.0), 1.0), True)}.items():
        rng = np.random.default_rng(1); res = []
        for rep in range(200):
            u = qmc.Sobol(4, scramble=True, seed=rep).random(256)
            P = [lo[i] + u[:, i] * (hi[i] - lo[i]) for i in range(4)]
            y = truth(*P, Dgain=dg) + rng.normal(0, 3.3, 256)
            ww, *_ = np.linalg.lstsq(ff(*P, useD), y, rcond=None)
            m = np.all([(gr[i] >= lo[i] - 1e-9) & (gr[i] <= hi[i] + 1e-9) for i in range(4)], 0)
            G = gr[:, m]; j = (ff(*G, useD) @ ww).argmax()
            res.append(truth(*[np.array([x]) for x in G[:, j]], Dgain=dg)[0])
        print(f'{scen:24s} {name:14s} ceiling {ceil:.2f}  true at fitted peak {np.mean(res):.2f} (p10 {np.percentile(res,10):.2f})')
