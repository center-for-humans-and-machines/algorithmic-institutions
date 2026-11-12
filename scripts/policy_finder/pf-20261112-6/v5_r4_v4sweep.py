"""v5: round-4 effect on v4's sweep, by target's contribution class x per-member dose.
Treatment of class k / dose bin j: total points dealt at round 4 to own valid members whose c4 is in class k with a
per-member dose in bin j, under the game's design point. E[.|history] = mean over the design points (restricted to
the same subset of points). Residual = as-good-as-random. Clean subset: last_b == 1 points (no later punishment)."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from load_sweep import load
base = 'plots/simulation/policy_finder/pf-20261112-6-v4_sweep/'
if not os.path.exists('/tmp/v4sweep.npz'):
    np.savez('/tmp/v4sweep.npz', **load(base))
D = np.load('/tmp/v4sweep.npz')
c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
pts = {q['name']: q['params'] for q in json.load(open(base + 'sweep.json'))['points']}
keys = np.array(sorted(pts))
PAR = np.array([[pts[k]['last_b'], pts[k]['thr'], pts[k]['amt'], pts[k]['lo'], pts[k]['amt_fr']] for k in keys])
pid = np.searchsorted(keys, name); assert (keys[pid] == name).all()
N = len(c); b = 4
mine = g == mg[:, None, None]
op = (1.6 * c * mine - p * mine).sum(1); oo = (1.6 * c * ~mine - p * ~mine).sum(1)
size = mine.sum(1)
c4 = c[:, :, b]; here = mine[:, :, b] & (v[:, :, b] > 0)
def dose(par):
    lb, thr, amt, lo, afr = par
    return np.where(c4 <= lo, np.floor(afr), np.where(c4 <= thr, np.floor(amt), 0.)) * here
chk = np.array([np.allclose(dose(PAR[pid[i]])[i], p[i, :, b] * mine[i, :, b]) for i in range(0, N, 97)])
print('check', chk.mean(), 'N', N)
classes = [(0, 1), (2, 4), (5, 7), (8, 10), (11, 20)]
dbins = [(1, 4), (4, 8), (8, 17)]
def feats(par):
    d = dose(par); out = []
    for lo_, hi_ in classes:
        cm = (c4 >= lo_) & (c4 <= hi_)
        for a, bb in dbins:
            out.append((d * cm * (d >= a) * (d < bb)).sum(1))
    return np.stack(out, 1)
labs = [f'c{a}-{b_} d{x}-{y}' for a, b_ in classes for x, y in dbins]
def run(sel_pts, title, outs):
    W = np.zeros((N, len(labs)), np.float32); F = None
    idx = np.where(sel_pts)[0]
    for i in idx: W += feats(PAR[i])
    E = W / len(idx)
    gsel = sel_pts[pid]
    P = np.zeros((N, len(labs)), np.float32)
    for i in idx:
        m = pid == i
        P[m] = feats(PAR[i])[m]
    R = (P - E)[gsel]; Eg = E[gsel]
    print('==', title, 'games', gsel.sum())
    for yl, y in outs.items():
        y = y[gsel]
        X = np.column_stack([np.ones(len(y)), R, Eg])
        bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (e ** 2).mean())
        print(' ', yl)
        for i, l in enumerate(labs):
            print(f'    {l:12s} {bb[i+1]:+.4f} ({se[i+1]:.4f})  meanP {P[gsel][:, i].mean():.2f}')
outs = {'lead rest': (op[:, b:] - oo[:, b:]).sum(1) / 24,
        'size r8': size[:, 8].astype(float),
        'own c/round 5-23': (c * mine)[:, :, 5:].sum(1).mean(1)}
run(PAR[:, 0] == 1, 'last_b=1', outs)
run(np.ones(len(keys), bool), 'all points, lead rounds 4-7 only + size r8', {'lead 4-7': (op[:, 4:8] - oo[:, 4:8]).sum(1) / 24, 'size r8': size[:, 8].astype(float), 'lead rest': outs['lead rest']})
