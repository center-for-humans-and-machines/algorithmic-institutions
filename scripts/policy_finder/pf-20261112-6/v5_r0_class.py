"""v5: round 0 on v2's sweep by target class (points with n_blocks <= 1: punish at round 0 only, or never)."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from load_sweep import load
ver = 'pf-20261112-6-v2'; base = f'plots/simulation/policy_finder/{ver}_sweep/'
f = '/tmp/v2sweep.npz'
if not os.path.exists(f): np.savez(f, **load(base))
D = np.load(f); c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
pts = {q['name']: q['params'] for q in json.load(open(base + 'sweep.json'))['points']}
keys = np.array(sorted(pts)); PAR = np.array([[pts[k][x] for x in ['n_blocks', 'thr', 'streak', 'amt']] for k in keys])
pid = np.searchsorted(keys, name)
idx = np.where(PAR[:, 0] <= 1)[0]; gs = np.isin(pid, idx)
c, p, g, v, mg, pid = c[gs], p[gs], g[gs], v[gs], mg[gs], pid[gs]
mine = g == mg[:, None, None]; size = mine.sum(1); N = len(c)
here = mine[:, :, 0] & (v[:, :, 0] > 0); c0 = c[:, :, 0]
Ds = np.stack([np.floor(PAR[i, 3]) * (c0 <= PAR[i, 1]) * here * (PAR[i, 0] >= 1) for i in idx], 1)
act = p[:, :, 0] * mine[:, :, 0]
print('games', N, 'check', np.mean(np.abs(Ds[np.arange(N), np.searchsorted(idx, pid)] - act).max(1) < 1e-6))
op = (1.6 * c * mine - p * mine).sum(1); oo = (1.6 * c * ~mine - p * ~mine).sum(1)
outs = {'lead': (op - oo).sum(1) / 24, 's4': size[:, 4].astype(float), 'ownc': (c * mine)[:, :, 1:].sum(1).mean(1)}
cl = [(0, 2), (3, 5), (6, 8)]
R, E = [], []
for a, z in cl:
    cm = (c0 >= a) & (c0 <= z)
    for fn in (lambda d: d, lambda d: d ** 2 / 10):
        P = (fn(act) * cm).sum(1); Ex = (fn(Ds) * cm[:, None]).sum(2).mean(1); R.append(P - Ex); E.append(Ex)
X = np.column_stack([np.ones(N)] + R + E)
for k, y in outs.items():
    bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb; V = np.linalg.pinv(X.T @ X) * (e ** 2).mean()
    line = k
    for i, (a, z) in enumerate(cl):
        for dd in (5, 9, 11):
            gv = np.zeros(len(bb)); gv[1 + 2 * i] = dd; gv[2 + 2 * i] = dd ** 2 / 10
            line += f' c{a}-{z} d{dd} {gv @ bb:+.2f}({np.sqrt(gv @ V @ gv):.2f})'
    print(line)
