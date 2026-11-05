"""v6: pool the v4 and v5 clone sweeps (same structure) in common coordinates
(dose at c = 0, 10, 16 after the clamp, horizon, lthr, lstop) and fit a GP."""
import json, numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

B = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'


def dose(c, pm, thr, g):
    fr = np.clip((thr - c) / thr, 0, 1)
    return np.minimum(30, pm * fr ** g) * (c < 20)


def feats(v, p):
    if v == 'v4':
        pm, thr, g = p['pmax'], p['thr'], p['gamma']
    else:
        pm, thr, g = p['pmax'], 20, np.log2(p['pmax'] / p['d10'])
    return [dose(0, pm, thr, g), dose(10, pm, thr, g), dose(16, pm, thr, g),
            p['horizon'], p['lthr'], p['lstop']]


rows = []
for v, name in [('v4', 'pf-20261105-3-v4'), ('v5', 'pf-20261105-3-v5')]:
    s = json.load(open(B + name + '_sweep/sweep.json'))
    for q in s['points']:
        rows.append((v, feats(v, q['params']), q['pool'], q['pool_se']))
X = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows]); se = np.array([r[3] for r in rows])
src = np.array([r[0] for r in rows])
print('n', len(y), 'mean by version', {v: y[src == v].mean().round(1) for v in ['v4', 'v5']})
lo, hi = X.min(0), X.max(0)
Z = (X - lo) / (hi - lo)
k = C(30.0) * RBF(length_scale=[0.5] * 6, length_scale_bounds=(0.05, 20)) + WhiteKernel(5, (0.5, 50))
gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=4, random_state=0).fit(Z, y)
print(gp.kernel_)
names = ['d0', 'd10', 'd16', 'horizon', 'lthr', 'lstop']
print('ranges', dict(zip(names, zip(lo.round(2), hi.round(2)))))
# random search for the peak
rng = np.random.default_rng(0)
cand = rng.random((200000, 6))
for j in [3, 4, 5]:
    cand[:, j] = np.round(cand[:, j] * (hi[j] - lo[j])) / (hi[j] - lo[j])
m, s = gp.predict(cand, return_std=True)
i = m.argmax(); pk = cand[i]
print('peak', dict(zip(names, (pk * (hi - lo) + lo).round(2))), m[i].round(2), s[i].round(2))
# profiles along each axis at the peak
for j, nm in enumerate(names):
    grid = np.linspace(0, 1, 9)
    P = np.tile(pk, (9, 1)); P[:, j] = grid
    mm, ss = gp.predict(P, return_std=True)
    print(nm, ' '.join(f'{g:.1f}:{a:.1f}' for g, a in zip(grid * (hi[j] - lo[j]) + lo[j], mm)))
# predictions at the finals
for v, name in [('v4', 'pf-20261105-3-v4'), ('v5', 'pf-20261105-3-v5')]:
    f = json.load(open(B + name + '_sweep/final/sweep.json'))
    for q in f['points']:
        z = (np.array(feats(v, q['params'])) - lo) / (hi - lo)
        print(q['name'], np.round(feats(v, q['params']), 1), 'gp', gp.predict(z[None])[0].round(1), 'final', round(q['pool'], 1))

# feasible candidates: v4's form with thr up to 40
print('--- feasible search')
n = 300000
pm = rng.uniform(22, 40, n); thr = rng.uniform(18, 40, n); g = rng.uniform(1.2, 6, n)
hz = rng.integers(3, 11, n); lt = rng.integers(0, 9, n); ls = rng.integers(8, 23, n)
F = np.stack([dose(0, pm, thr, g), dose(10, pm, thr, g), dose(16, pm, thr, g), hz, lt, ls], 1)
ok = np.all((F >= lo) & (F <= hi), 1)
F, pm, thr, g = F[ok], pm[ok], thr[ok], g[ok]
mm, ss = gp.predict((F - lo) / (hi - lo), return_std=True)
o = np.argsort(-mm)[:15]
for i in o:
    print(np.round(F[i], 2), f'pmax {pm[i]:.1f} thr {thr[i]:.1f} g {g[i]:.2f}', mm[i].round(2), ss[i].round(2))
# score vs d16 at d10 in 6-7.5 and horizon 5-8 lthr 2-5 lstop 13-16
sel = (F[:, 1] > 5.5) & (F[:, 1] < 7.5) & (F[:, 3] >= 5) & (F[:, 3] <= 8) & (F[:, 4] >= 2) & (F[:, 4] <= 5) & (F[:, 5] >= 13) & (F[:, 5] <= 16)
for a, b in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4)]:
    s2 = sel & (F[:, 2] >= a) & (F[:, 2] < b)
    print('d16', a, b, s2.sum(), mm[s2].mean().round(2) if s2.sum() else None)
