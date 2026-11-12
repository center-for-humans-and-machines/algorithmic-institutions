"""v7: round 8 on v4's sweep, split by whether the target was already hit at round 4.
Points with last_b in {1, 2}: same round-4 rule, round-8 hit only for last_b = 2. Given the history up to round 8
the posterior over design points is uniform over the consistent ones (dynamics depend on params only through
punishment), so E[P_8 | history] is the mean over them; regress outcomes on (actual - expected) per class."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from load_sweep import load
base = 'plots/simulation/policy_finder/pf-20261112-6-v4_sweep/'
D = load(base)
c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']; del D
pts = {q['name']: q['params'] for q in json.load(open(base + 'sweep.json'))['points']}
keys = np.array(sorted(pts))
PAR = np.array([[pts[k]['last_b'], pts[k]['thr'], pts[k]['amt'], pts[k]['lo'], pts[k]['amt_fr']] for k in keys])
pid = np.searchsorted(keys, name)
gsel = np.isin(PAR[pid, 0], [1, 2])
c, p, g, v, mg, pid = c[gsel], p[gsel], g[gsel], v[gsel], mg[gsel], pid[gsel]
N = len(c); mine = g == mg[:, None, None]
op = (1.6 * c * mine - p * mine).sum(1); oo = (1.6 * c * ~mine - p * ~mine).sum(1)
size = mine.sum(1)
idx = np.where(np.isin(PAR[:, 0], [1, 2]))[0]
def dose(par, r):
    lb, thr, amt, lo, afr = par
    if r > 4 * lb: return np.zeros((N, 8))
    cr = c[:, :, r]; here = mine[:, :, r] & (v[:, :, r] > 0)
    return np.where(cr <= lo, np.floor(afr), np.where(cr <= thr, np.floor(amt), 0.)) * here
b = 8
cons = np.ones((N, len(idx)), bool)
obs4 = p[:, :, 4] * mine[:, :, 4]
for j, i in enumerate(idx):
    cons[:, j] &= (np.abs(dose(PAR[i], 4) - obs4) < 1e-6).all(1)
print('games', N, 'consistent pts', cons.sum(1).mean(), 'actual consistent', cons[np.arange(N), np.searchsorted(idx, pid)].mean())
cb = c[:, :, b]
hit4 = obs4 > 0
new8 = mine[:, :, 8] & ~mine[:, :, 7]
was4 = mine[:, :, 4]
low = (cb <= 5)
classes = {'c0-5 hit@4': low & hit4, 'c0-5 nothit stay(in@4)': low & ~hit4 & was4 & ~new8,
           'c0-5 nothit new@8': low & ~hit4 & new8, 'c0-5 nothit other': low & ~hit4 & ~was4 & ~new8,
           'c6-10': (cb >= 6) & (cb <= 10), 'c11+': cb >= 11}
for k, m in classes.items(): print(k, 'members per game', (m & mine[:, :, 8]).sum(1).mean().round(3))
def feats(d): return np.stack([(d * m).sum(1) for m in classes.values()], 1)
W = np.stack([feats(dose(PAR[i], b)) for i in idx], 1)
E = (W * cons[..., None]).sum(1) / np.maximum(cons.sum(1), 1)[:, None]
P = feats(p[:, :, b] * mine[:, :, b])
R = P - E
outs = {'lead rest': (op[:, b:] - oo[:, b:]).sum(1) / 24, 'size r12': size[:, 12].astype(float),
        'size r16': size[:, 16].astype(float), 'own c/round rest': (c * mine)[:, :, b + 1:].sum(1).mean(1)}
for yl, y in outs.items():
    X = np.column_stack([np.ones(N), R, E, size[:, b]])
    bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (e ** 2).mean())
    print(yl, ' | '.join(f'{k} {bb[j+1]:+.4f}({se[j+1]:.4f})' for j, k in enumerate(classes)))
# average dose given a hit at round 8 by class (actual)
for j, k in enumerate(classes):
    m = list(classes.values())[j] & mine[:, :, 8] & (p[:, :, 8] > 0)
    print(k, 'mean dose when hit', p[:, :, 8][m].mean().round(1) if m.any() else None, 'n hits', m.sum())
