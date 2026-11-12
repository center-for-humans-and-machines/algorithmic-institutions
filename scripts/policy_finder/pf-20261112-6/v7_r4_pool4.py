"""v7 (copy of v6_r4_pool3.py plus the v6 sweep as source 3): pool the clean round-4-only design points of the v3 (last_b=1), v4 (last_b=1) and v5 (all) sweeps.
Per game and player: actual round-4 dose, its design expectation (and of dose^2/10) given the history.
Saves a compact npz (/tmp/v7_pool4.npz) for the fits in v6_r4_fits.py."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from load_sweep import load
b = 4
out = {k: [] for k in ['act', 'e1', 'e2', 'c4', 'c3', 'cpast', 'here', 'size4', 'lead', 's8', 's12', 'ownc', 'src',
                       'oth', 'osize4', 'seat', 'lead_r', 'new4']}

def prep(ver, cols, dosefn, clean, j):
    base = f'plots/simulation/policy_finder/{ver}_sweep/'
    D = load(base)
    c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
    del D
    pts = {q['name']: q['params'] for q in json.load(open(base + 'sweep.json'))['points']}
    keys = np.array(sorted(pts)); PAR = np.array([[pts[k][x] for x in cols] for k in keys])
    pid = np.searchsorted(keys, name)
    idx = np.where(clean(PAR))[0]; gs = np.isin(pid, idx)
    c, p, g, v, mg, pid = c[gs], p[gs], g[gs], v[gs], mg[gs], pid[gs]
    mine = g == mg[:, None, None]; size = mine.sum(1)
    here = mine[:, :, b] & (v[:, :, b] > 0)
    e1 = np.zeros(here.shape); e2 = np.zeros(here.shape)
    for i in idx:
        d = dosefn(PAR[i], c, here, size); e1 += d; e2 += d ** 2 / 10
    e1 /= len(idx); e2 /= len(idx)
    act = p[:, :, b] * mine[:, :, b]
    own = dosefn(None, c, here, size, P=PAR[pid])
    print(ver, 'games', len(c), 'check', np.mean(np.abs(own - act).max(1) < 1e-6))
    op = (1.6 * c * mine - p * mine).sum(1); oo = (1.6 * c * ~mine - p * ~mine).sum(1)
    m4 = mine[:, :, b]; s = (c[:, :, b] * m4).sum(1, keepdims=True); n = m4.sum(1, keepdims=True)
    for k, x in dict(act=act, e1=e1, e2=e2, c4=c[:, :, b], c3=c[:, :, 3], cpast=c[:, :, :4].mean(2), here=here,
                     size4=size[:, b], lead=(op[:, b:] - oo[:, b:]).sum(1) / 24, s8=size[:, 8], s12=size[:, 12],
                     ownc=(c * mine)[:, :, 5:].sum(1).mean(1), src=np.full(len(c), j),
                     oth=(s - c[:, :, b]) / np.maximum(n - 1, 1), osize4=8 - size[:, b], seat=mg,
                     lead_r=(op - oo) / 24, new4=m4 & ~mine[:, :, 3]).items():
        out[k].append(x.astype(np.float32) if x.dtype != bool else x)

def d5(par, c, here, size, P=None):
    P = P if P is not None else par[None]; cr = c[:, :, b]
    return np.where((cr >= P[:, 0:1]) & (cr <= P[:, 1:2]), np.floor(P[:, 2:3]), 0.) * here
def d4(par, c, here, size, P=None):
    P = P if P is not None else par[None]; cr = c[:, :, b]
    lb, thr, amt, lo, afr = [P[:, k:k + 1] for k in range(5)]
    return np.where(cr <= lo, np.floor(afr), np.where(cr <= thr, np.floor(amt), 0.)) * here
def d3(par, c, here, size, P=None):
    P = P if P is not None else par[None]
    nmin, lb, thr, st, amt = [P[:, k:k + 1] for k in range(5)]
    low = np.ones(here.shape, bool)
    for k in range(3): low &= (c[:, :, b - k] <= thr) | (k >= st)
    return np.floor(amt) * (low & here) * (size[:, b:b + 1] >= nmin)

prep('pf-20261112-6-v6', ['lo', 'hi', 'amt'], d5, lambda P: np.ones(len(P), bool), 3)
prep('pf-20261112-6-v5', ['lo', 'hi', 'amt'], d5, lambda P: np.ones(len(P), bool), 0)
prep('pf-20261112-6-v4', ['last_b', 'thr', 'amt', 'lo', 'amt_fr'], d4, lambda P: P[:, 0] == 1, 1)
prep('pf-20261112-6-v3', ['n_min', 'last_b', 'thr', 'streak', 'amt'], d3, lambda P: P[:, 1] == 1, 2)
np.savez('/tmp/v7_pool4.npz', **{k: np.concatenate(v) for k, v in out.items()})
