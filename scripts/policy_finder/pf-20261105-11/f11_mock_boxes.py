"""v6: mock sweeps of 3D boxes (S fixed 8). Truth: (a) the pooled v2-v4 quadratic of e6; (b) the index-FE quadratic of
f6 (ridge to the low-T/high-A corner). Noise sd 3.3, 256 Sobol points, quadratic fit, true value at the fitted peak."""
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
tA = lambda T, A, lB: feats(T, A, lB, np.full_like(T, 8.0)) @ w
bq = np.load('/tmp/pf6/crn_beta.npy')
def XF(T, A, lB, Sv, N):
    g = np.minimum(N, 12)
    return np.column_stack([np.ones_like(T), T, A, lB, Sv, T*T, A*A, lB*lB, Sv*Sv, T*A, T*lB, A*lB, T*Sv, lB*Sv, A*Sv, g, g*g])
tB = lambda T, A, lB: XF(T, A, lB, np.full_like(T, 8.0), np.full_like(T, 12.0)) @ bq
gr = np.array(np.meshgrid(np.arange(11, 16.01, .2), np.arange(0, 6.01, .2), np.linspace(np.log(1.1), np.log(3.2), 30), indexing='ij')).reshape(3, -1)
boxes = {'v5 box T[11.5,15] A[1,5] B[1.3,3]': ((11.5, 1, np.log(1.3)), (15, 5, np.log(3.0))),
         'T[12,14.8] A[1,4] B[1.4,2.4]': ((12, 1, np.log(1.4)), (14.8, 4, np.log(2.4))),
         'T[11.5,14.5] A[1,5] B[1.4,2.6]': ((11.5, 1, np.log(1.4)), (14.5, 5, np.log(2.6))),
         'T[11.8,14.8] A[0.8,4.6] B[1.35,2.6]': ((11.8, 0.8, np.log(1.35)), (14.8, 4.6, np.log(2.6)))}
for tn, tr in [('pooled quad', tA), ('FE quad', tB)]:
    ceil = tr(*gr).max()
    for name, (lo, hi) in boxes.items():
        rng = np.random.default_rng(2); res = []
        for rep in range(300):
            u = qmc.Sobol(3, scramble=True, seed=rep).random(256)
            P = [lo[i] + u[:, i] * (hi[i] - lo[i]) for i in range(3)]
            y = tr(*P) + rng.normal(0, 3.3, 256)
            F = lambda T, A, lB: np.column_stack([np.ones_like(T), T, T**2, lB, lB**2, T*lB, A, A**2, A*lB, A*T])
            ww, *_ = np.linalg.lstsq(F(*P), y, rcond=None)
            m = np.all([(gr[i] >= lo[i] - 1e-9) & (gr[i] <= hi[i] + 1e-9) for i in range(3)], 0)
            G = gr[:, m]; j = (F(*G) @ ww).argmax()
            res.append(tr(*[np.array([x]) for x in G[:, j]])[0])
        print(f'{tn:12s} {name:38s} ceiling {ceil:.2f} true at fitted peak {np.mean(res):.2f} (p10 {np.percentile(res,10):.2f})')
