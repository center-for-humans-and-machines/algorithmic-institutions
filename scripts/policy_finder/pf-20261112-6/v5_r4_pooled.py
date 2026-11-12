"""v5: round-4 effect per target contribution value, pooling the clean (round-4-only) points of the v3 and v4
sweeps. Treatment per class: points dealt (linear) and points^2/10 (curvature), residualized on the mean over the
same sweep's clean points. Also by own size at round 4 and by the target's mean contribution in rounds 0-3."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from load_sweep import load
b = 4
def prep(ver, cols, dosefn, clean):
    base = f'plots/simulation/policy_finder/{ver}_sweep/'
    f = f'/tmp/{ver.split("-")[-1]}sweep.npz'
    if not os.path.exists(f): np.savez(f, **load(base))
    D = np.load(f)
    c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
    pts = {q['name']: q['params'] for q in json.load(open(base + 'sweep.json'))['points']}
    keys = np.array(sorted(pts)); PAR = np.array([[pts[k][x] for x in cols] for k in keys])
    pid = np.searchsorted(keys, name)
    idx = np.where(clean(PAR))[0]; gs = np.isin(pid, idx)
    c, p, g, v, mg, pid = c[gs], p[gs], g[gs], v[gs], mg[gs], pid[gs]
    mine = g == mg[:, None, None]; size = mine.sum(1)
    here = mine[:, :, b] & (v[:, :, b] > 0)
    Ds = np.stack([dosefn(PAR[i], c, here, size) for i in idx], 1)  # n, pts, 8
    act = p[:, :, b] * mine[:, :, b]
    print(ver, 'games', len(c), 'check', np.mean(np.abs(Ds[np.arange(len(c)), np.searchsorted(idx, pid)] - act).max(1) < 1e-6))
    op = (1.6 * c * mine - p * mine).sum(1); oo = (1.6 * c * ~mine - p * ~mine).sum(1)
    return dict(c=c, mine=mine, size=size, act=act, Ds=Ds, lead=(op[:, b:] - oo[:, b:]).sum(1) / 24,
                s8=size[:, 8].astype(float), ownc=(c * mine)[:, :, 5:].sum(1).mean(1), src=np.full(len(c), ver[-2:]))
def d4(par, c, here, size):
    lb, thr, amt, lo, afr = par; cr = c[:, :, b]
    return np.where(cr <= lo, np.floor(afr), np.where(cr <= thr, np.floor(amt), 0.)) * here
def d3(par, c, here, size):
    nmin, lb, thr, st, amt = par
    low = np.ones(here.shape, bool)
    for k in range(int(st)): low &= c[:, :, b - k] <= thr
    return np.floor(amt) * (low & here) * (size[:, b] >= nmin)[:, None]
A = prep('pf-20261112-6-v4', ['last_b', 'thr', 'amt', 'lo', 'amt_fr'], d4, lambda P: P[:, 0] == 1)
B = prep('pf-20261112-6-v3', ['n_min', 'last_b', 'thr', 'streak', 'amt'], d3, lambda P: P[:, 1] == 1)
classes = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10), (11, 20)]
def design(S, mask=None):
    cb = S['c'][:, :, b]
    cols_R, cols_E = [], []
    for a, z in classes:
        cm = (cb >= a) & (cb <= z)
        if mask is not None: cm = cm & mask
        for f in (lambda d: d, lambda d: d ** 2 / 10):
            P = (f(S['act']) * cm).sum(1); E = (f(S['Ds']) * cm[:, None]).sum(2).mean(1)
            cols_R.append(P - E); cols_E.append(E)
    return np.array(cols_R).T, np.array(cols_E).T
def fit(sel=None, mask_fn=None, title=''):
    Xs, ys = [], {k: [] for k in ['lead', 's8', 'ownc']}
    for j, S in enumerate([A, B]):
        R, E = design(S, None if mask_fn is None else mask_fn(S))
        s = np.ones(len(R), bool) if sel is None else sel(S)
        src = np.full((s.sum(), 1), j)
        Xs.append(np.column_stack([R[s], E[s], src, S['size'][s, b]]))
        for k in ys: ys[k].append(S[k][s])
    X = np.column_stack([np.ones(sum(len(x) for x in Xs)), np.concatenate(Xs)])
    print('==', title, 'games', len(X))
    for k in ys:
        y = np.concatenate(ys[k]); bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb
        V = np.linalg.pinv(X.T @ X) * (e ** 2).mean()
        line = f'  {k:5s}'
        for i, (a, z) in enumerate(classes):
            l, q = 1 + 2 * i, 2 + 2 * i
            # marginal per point at dose 5 and 13, and total effect of a dose of 5 / 13
            for dd in (5, 13):
                gvec = np.zeros(len(bb)); gvec[l] = dd; gvec[q] = dd ** 2 / 10
                tot = gvec @ bb; se = np.sqrt(gvec @ V @ gvec)
                line += f' | c{a}-{z} d{dd} {tot:+.2f}({se:.2f})' if dd == 5 else f' d{dd} {tot:+.2f}({se:.2f})'
        print(line)
fit(title='all clean games, total effect of hitting one member with 5 / 13 points')
fit(sel=lambda S: S['size'][:, b] <= 4, title='own size <= 4 at round 4')
fit(sel=lambda S: S['size'][:, b] >= 5, title='own size >= 5 at round 4')
