# v4: per-opponent surfaces over v3's design (GP on the 256 points' per-opponent pools): the peak per
# opponent, its value, and each opponent's pool at the overall peak and at the other opponents' peaks.
import json, sys, numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
ver = sys.argv[1] if len(sys.argv) > 1 else 'pf-20261106-9-v3'
s = json.load(open(f'plots/simulation/policy_finder/{ver}_sweep/sweep.json'))
L = json.load(open(f'plots/simulation/policy_finder/{ver}_sweep/landscapes.json'))
names = list(L['params'])
lo = np.array([L['params'][k]['range'][0] for k in names]); hi = np.array([L['params'][k]['range'][1] for k in names])
islog = np.array([L['params'][k]['log'] for k in names])
def tr(P):
    P = np.array(P, float)
    P = np.where(islog, np.log(P), P)
    a = np.where(islog, np.log(lo), lo); b = np.where(islog, np.log(hi), hi)
    return (P - a) / (b - a)
X = tr([[q['params'][k] for k in names] for q in s['points']])
rng = np.random.default_rng(0)
cand = rng.random((60000, len(names)))
gps = {}
for o in s['opponents'] + ['all']:
    y = np.array([q['pools'][o]['pool'] if o != 'all' else q['pool'] for q in s['points']])
    gp = GaussianProcessRegressor(C(10) * RBF(np.ones(len(names)) * 0.5, (0.05, 20)) + WhiteKernel(1), normalize_y=True).fit(X, y)
    gps[o] = gp
pk = {}
for o, gp in gps.items():
    m = gp.predict(cand); i = m.argmax(); pk[o] = cand[i]
    a = np.where(islog, np.log(lo), lo); b = np.where(islog, np.log(hi), hi)
    raw = a + cand[i] * (b - a); raw = np.where(islog, np.exp(raw), raw)
    print(o, f'peak {m[i]:.1f}', dict(zip(names, np.round(raw, 2))), 'ls', np.round(gp.kernel_.k1.k2.length_scale, 2))
print('cross: rows = opponent surface, cols = evaluated at peak of')
for o, gp in gps.items():
    print(o, ' '.join(f'{k}:{gp.predict(pk[k][None])[0]:.1f}' for k in pk))
# profiles per opponent through the overall peak, for quiet and k
a = np.where(islog, np.log(lo), lo); b = np.where(islog, np.log(hi), hi)
for name in ['quiet', 'k', 'window', 's0', 'cap']:
    j = names.index(name)
    grid = np.linspace(0, 1, 7)
    print('profile', name, [round(float(np.exp(a[j] + x * (b[j] - a[j])) if islog[j] else a[j] + x * (b[j] - a[j])), 2) for x in grid])
    for o, gp in gps.items():
        Z = np.repeat(pk['all'][None], len(grid), 0); Z[:, j] = grid
        print('  ', o, np.round(gp.predict(Z), 1))
