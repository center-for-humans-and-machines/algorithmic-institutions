"""v5: rounds 8 and 12 on v4's sweep. Given the history up to round b, the posterior over design points is uniform
over the points consistent with every earlier punishment observed (the dynamics depend on params only through
punishment). E[P_b | history] = mean of the would-be round-b punishment over consistent points. For b=8 only
points with last_b == 2 (so nothing after 8), for b=12 points with last_b == 3."""
import numpy as np, json
D = np.load('/tmp/v4sweep.npz')
c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
base = 'plots/simulation/policy_finder/pf-20261112-6-v4_sweep/'
pts = {q['name']: q['params'] for q in json.load(open(base + 'sweep.json'))['points']}
keys = np.array(sorted(pts))
PAR = np.array([[pts[k]['last_b'], pts[k]['thr'], pts[k]['amt'], pts[k]['lo'], pts[k]['amt_fr']] for k in keys])
pid = np.searchsorted(keys, name)
N = len(c)
mine = g == mg[:, None, None]
op = (1.6 * c * mine - p * mine).sum(1); oo = (1.6 * c * ~mine - p * ~mine).sum(1)
size = mine.sum(1)
def dose(par, r):
    lb, thr, amt, lo, afr = par
    if r > 4 * lb: return np.zeros((N, 8))
    cr = c[:, :, r]; here = mine[:, :, r] & (v[:, :, r] > 0)
    return np.where(cr <= lo, np.floor(afr), np.where(cr <= thr, np.floor(amt), 0.)) * here
classes = [(0, 1), (2, 4), (5, 7), (8, 10), (11, 20)]
for b, lbsel in [(8, 2), (12, 3)]:
    idx = np.where(PAR[:, 0] == lbsel)[0]
    gsel = PAR[pid, 0] == lbsel
    cons = np.ones((N, len(idx)), bool)
    for r in range(4, b, 4):
        obs = p[:, :, r] * mine[:, :, r]
        for j, i in enumerate(idx):
            cons[:, j] &= (np.abs(dose(PAR[i], r) - obs) < 1e-6).all(1)
    cb = c[:, :, b]
    def feats(d):
        return np.stack([(d * ((cb >= a) & (cb <= z))).sum(1) for a, z in classes], 1)
    W = np.stack([feats(dose(PAR[i], b)) for i in idx], 1)  # N, npts, K
    E = (W * cons[..., None]).sum(1) / np.maximum(cons.sum(1), 1)[:, None]
    P = feats(p[:, :, b] * mine[:, :, b])
    print(f'== b={b} games {gsel.sum()}, consistent pts mean {cons[gsel].sum(1).mean():.1f}, actual consistent {cons[gsel, np.searchsorted(idx, pid[gsel])].mean():.3f}')
    R = (P - E)[gsel]
    outs = {'lead rest': (op[:, b:] - oo[:, b:]).sum(1) / 24, f'size r{b+4}': size[:, b + 4].astype(float),
            'own c/round rest': (c * mine)[:, :, b + 1:].sum(1).mean(1)}
    for yl, y in outs.items():
        y = y[gsel]
        X = np.column_stack([np.ones(len(y)), R, E[gsel], size[gsel, b]])
        bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (e ** 2).mean())
        print(' ', yl, '  '.join(f'c{a}-{z} {bb[k+1]:+.4f}({se[k+1]:.4f})' for k, (a, z) in enumerate(classes)))
