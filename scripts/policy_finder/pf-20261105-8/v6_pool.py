"""v6: pool all sweeps v1-v5 in 'what the rule does' features and fit one GP.

Features: main-phase punishment (clamped to 30, floored) at c = 0, 4, 8, 12, 16;
horizon; end (first round with no punishment at all); tail floor (players below it
are punished between horizon and end).
"""
import json, numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
B = 'plots/simulation/policy_finder/'
CS = np.array([0, 4, 8, 12, 16.])

def curve(dose, s, k):
    return np.floor(np.clip(dose * (np.clip(s - CS, 0, None) / s) ** k, 0, 30))

def feat(v, q):
    if v == 1:
        P = np.floor(np.clip(np.minimum(q['slope'] * np.clip(q['target'] - CS, 0, None), q['cap']), 0, 30))
        return list(P) + [q['horizon'], q['horizon'], 0.]
    if v == 2:
        return list(curve(q['dose'], q['standard'], q['curvature'])) + [q['horizon'], q['horizon'], 0.]
    if v == 3:
        e = q['horizon'] + q['tail']
        return list(curve(q['dose'], q['standard'], q['curvature'])) + [q['horizon'], e, q['floor'] if e > q['horizon'] else 0.]
    if v in (4, 5):
        e = max(q['end'], q['horizon'])
        return list(curve(q['dose'], q['standard'], q['rate'] * q['standard'])) + [q['horizon'], e, q['floor'] if e > q['horizon'] else 0.]

def load():
    rows = []
    for v in range(1, 6):
        name = 'pf-20261105-8' + ('' if v == 1 else f'-v{v}')
        for p in json.load(open(B + name + '_sweep/sweep.json'))['points']:
            rows.append(feat(v, p['params']) + [p['pool'], p['pool_se'], v])
    return np.array(rows)

names = ['P0', 'P4', 'P8', 'P12', 'P16', 'hor', 'end', 'floor']
if __name__ == '__main__':
    A = load(); X = A[:, :8]; y = A[:, 8]; ver = A[:, 10]
    print('n', len(y), 'per version', [int((ver == v).sum()) for v in range(1, 6)])
    keep = y > 610
    lo, hi = X[keep].min(0), X[keep].max(0); Z = (X - lo) / (hi - lo + 1e-9)
    gp = GaussianProcessRegressor(C(100) * RBF(np.ones(8) * .3, (1e-2, 1e2)) + WhiteKernel(10),
                                  normalize_y=True, n_restarts_optimizer=3, random_state=0)
    gp.fit(Z[keep], y[keep]); print(gp.kernel_)
    mu = gp.predict(Z); print('resid sd', np.std(y[keep] - mu[keep]), 'kept', keep.sum())
    for i in np.argsort(-mu)[:15]:
        print(int(ver[i]), X[i], 'true', round(y[i], 1), 'fit', round(mu[i], 1))
    import pickle; pickle.dump((gp, lo, hi), open('/tmp/x/gp6.pkl', 'wb'))
