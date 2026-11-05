"""v5: mock sweeps on the pooled v2-v4 quadratic (e1) as truth, noise sd 3.6 per point, 256 Sobol points.
Fit a quadratic (and a GP via sklearn when available) to the noisy points; report the true value at the fitted peak."""
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
truth = lambda T, A, lB, S: feats(T, A, lB, S) @ w
designs = {
    'v4': dict(T=(11.5, 16.5), A=(0.5, 4.5), lB=(np.log(1.15), np.log(3.0)), S=(5, 12)),
    'v5': dict(T=(11.5, 15.0), A=(1.0, 5.0), lB=(np.log(1.3), np.log(3.0)), S=(7, 7)),
    'v5S': dict(T=(11.5, 15.0), A=(1.0, 5.0), lB=(np.log(1.3), np.log(3.0)), S=(5, 10)),
}
gr = np.array(np.meshgrid(np.arange(11, 17.01, .25), np.arange(0, 5.01, .25), np.linspace(0, np.log(3.2), 30), np.arange(5, 13), indexing='ij')).reshape(4, -1)
print('ceiling', truth(*gr).max().round(2))
rng = np.random.default_rng(0)
for name, d in designs.items():
    res, near = [], []
    for rep in range(200):
        u = qmc.Sobol(4, scramble=True, seed=rep).random(256)
        T = d['T'][0] + u[:, 0] * (d['T'][1] - d['T'][0]); A = d['A'][0] + u[:, 1] * (d['A'][1] - d['A'][0])
        lB = d['lB'][0] + u[:, 2] * (d['lB'][1] - d['lB'][0]); S = np.floor(d['S'][0] + u[:, 3] * (d['S'][1] - d['S'][0] + 1)).clip(d['S'][0], d['S'][1])
        tv = truth(T, A, lB, S); y = tv + rng.normal(0, 3.6, 256)
        F = feats(T, A, lB, S)
        if d['S'][0] == d['S'][1]: F = F[:, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]]
        ww, *_ = np.linalg.lstsq(F, y, rcond=None)
        m = (gr[0] >= d['T'][0]) & (gr[0] <= d['T'][1]) & (gr[1] >= d['A'][0]) & (gr[1] <= d['A'][1]) & (gr[2] >= d['lB'][0]) & (gr[2] <= d['lB'][1]) & (gr[3] >= d['S'][0]) & (gr[3] <= d['S'][1])
        G = gr[:, m]; FG = feats(*G)
        if d['S'][0] == d['S'][1]: FG = FG[:, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]]
        k = (FG @ ww).argmax()
        res.append(truth(*[np.array([x]) for x in G[:, k]])[0]); near.append((tv > truth(*gr).max() - 2).mean())
    print(name, 'true value at fitted peak: mean', np.mean(res).round(2), 'p10', np.percentile(res, 10).round(2), ' share of points within 2 of ceiling', np.mean(near).round(2))
