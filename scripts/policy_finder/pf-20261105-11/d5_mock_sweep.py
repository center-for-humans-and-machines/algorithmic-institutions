"""v4: mock sweeps. Truth = the pooled v2+v3 quadratic (d1, N = never give up); noise sd 3.7 per
point as in the real sweeps. Fit a quadratic to 256 noisy points of each design, take its peak,
and score that peak on the truth. Compares v2's, v3's (N dropped) and v4's ranges."""
import json, sys, numpy as np
sys.path.insert(0, 'src')
from aimanager.manager.rule import sobol_design, read_rule
w = np.array([250.200, 45.994, -1.399, 146.071, -23.990, -7.740, 25.149, -0.684, 1.585, -0.057, 1.534, -0.050, -0.080, -4.298, -1.418])
def feats(T, A, B, S, N=10):
    lB = np.log(B)
    return np.column_stack([np.ones_like(T), T, T**2, lB, lB**2, T*lB, A, A**2, S, S**2, N + 0*T, N**2 + 0*T, S*N, A*lB, A*T])
def truth(T, A, B, S): return feats(T, A, B, S) @ w
def fitfe(T, A, B, S):
    lB = np.log(B)
    return np.column_stack([np.ones_like(T), T, T**2, lB, lB**2, T*lB, A, A**2, S, S**2, A*lB, A*T, S*T, S*lB, S*A])
import yaml
designs = {}
for v in ['pf-20261105-11-v2', 'pf-20261105-11-v3', 'pf-20261105-11-v4']:
    rule = yaml.safe_load(open(f'configs/managers/rule_based/{v}.yml'))
    pts = sobol_design(rule, 256)
    designs[v] = np.array([[q['T'], q['A'], q['B'], q['S']] for q in pts])
rng = np.random.default_rng(0)
print('truth max over a fine grid ~', truth(np.array([13.25]), np.array([2.5]), np.array([2.0]), np.array([7.]))[0].round(1))
for v, D in designs.items():
    T, A, B, S = D.T
    y0 = truth(T, A, B, S)
    print(v, 'ranges', D.min(0).round(2), D.max(0).round(2), 'share of points within 2 of the peak', (y0 > 647.8).mean().round(3))
    vals, vals_best = [], []
    for r in range(300):
        y = y0 + rng.normal(0, 3.7, len(y0))
        F = fitfe(T, A, B, S); ww, *_ = np.linalg.lstsq(F, y, rcond=None)
        # peak of the fit over the design's box (random candidates)
        C = rng.uniform(D.min(0), D.max(0), (20000, 4)); C[:, 2] = np.exp(rng.uniform(np.log(D[:, 2].min()), np.log(D[:, 2].max()), 20000)); C[:, 3] = np.round(C[:, 3])
        k = np.argmax(fitfe(*C.T) @ ww)
        vals.append(truth(*C[k:k+1].T)[0])
        vals_best.append(y0[np.argmax(y)])
    print(f'   fitted-peak truth mean {np.mean(vals):.2f} p10 {np.percentile(vals,10):.2f}; best-noisy-point truth {np.mean(vals_best):.2f}')
