"""v7: pool v4, v5, v6 clone sweeps (one structure, three parametrisations) in common
coordinates (shock dose after clamp+floor at c = 0, 5, 10, 13, 16, 18; horizon, lthr,
lstop) and remove the per-seed-slot effect (design index i plays the same games in
every version) by alternating a GP fit and slot means."""
import json, numpy as np, sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

B = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
CS = [0, 5, 10, 13, 16, 18]


def curve(v, p, c):
    c = np.asarray(c, float)
    if v == 'v4':
        fr = np.clip((p['thr'] - c) / p['thr'], 0, 1)
        d = p['pmax'] * fr ** p['gamma']
    elif v == 'v5':
        d = p['pmax'] * (1 - c / 20) ** np.log2(p['pmax'] / p['d10'])
    else:
        g = np.log2(p['pmax'] / p['d10']); k = np.log(p['d10'] / p['d16']) / np.log(2.5)
        d = np.where(c <= 10, p['pmax'] * np.clip(1 - c / 20, 0, 1) ** g,
                     p['d10'] * np.clip((20 - c) / 10, 0, 1) ** k)
    return np.floor(np.minimum(30, d)) * (c < 20)


def feats(v, p):
    return list(curve(v, p, CS)) + [p['horizon'], p['lthr'], p['lstop']]


def load():
    rows = []
    for v in ['v4', 'v5', 'v6']:
        s = json.load(open(B + f'pf-20261105-3-{v}_sweep/sweep.json'))
        for q in s['points']:
            idx = int(q['name'].split('_s')[-1])
            rows.append((v, idx, feats(v, q['params']), q['pool']))
    return rows


rows = load()
X = np.array([r[2] for r in rows], float); y = np.array([r[3] for r in rows]); slot = np.array([r[1] for r in rows])
src = np.array([r[0] for r in rows])
lo, hi = X.min(0), X.max(0); hi[hi == lo] += 1
Z = (X - lo) / (hi - lo)
names = [f'd{c}' for c in CS] + ['horizon', 'lthr', 'lstop']
se = np.zeros(256)
for it in range(4):
    yy = y - se[slot]
    k = C(30.0) * RBF(length_scale=[0.5] * Z.shape[1], length_scale_bounds=(0.05, 50)) + WhiteKernel(5, (0.1, 50))
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(Z, yy)
    res = y - gp.predict(Z)
    # slot effect: mean residual over the (up to 3) versions, shrunk
    sm = np.bincount(slot, res, 256) / np.maximum(np.bincount(slot, None, 256), 1)
    se = sm * 0.9
    print('iter', it, gp.kernel_, 'slot sd', sm.std().round(2), 'resid sd after slot', (res - se[slot]).std().round(2), flush=True)
np.save('/tmp/v7_slot.npy', se)
print('raw sd of y within version', {v: y[src == v].std().round(2) for v in ['v4', 'v5', 'v6']})
# top adjusted points
adj = y - se[slot]
o = np.argsort(-gp.predict(Z))[:15]
for i in o:
    print(src[i], slot[i], X[i].astype(int) if False else X[i], 'gp', gp.predict(Z[i:i + 1])[0].round(1), 'raw', y[i].round(1), 'adj', adj[i].round(1))
import pickle
pickle.dump((gp, lo, hi, names), open('/tmp/v7_gp.pkl', 'wb'))
