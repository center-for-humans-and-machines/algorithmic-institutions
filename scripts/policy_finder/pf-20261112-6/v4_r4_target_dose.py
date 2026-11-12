"""v4: round-4 effect by target contribution x per-member dose (v3 sweep, all points; design as v4_r4_dose.py)."""
import numpy as np, json
D = np.load('/tmp/v3sweep.npz')
c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
base = 'plots/simulation/policy_finder/pf-20261112-6-v3_sweep/'
pts = {q['name']: q['params'] for q in json.load(open(base + 'sweep.json'))['points']}
keys = np.array(sorted(pts))
PAR = np.array([[pts[k]['n_min'], pts[k]['last_b'], pts[k]['thr'], pts[k]['streak'], pts[k]['amt']] for k in keys])
pid = np.searchsorted(keys, name)
N = len(c); b = 4
mine = g == mg[:, None, None]
op = (1.6 * c * mine - p * mine).sum(1); oo = (1.6 * c * ~mine - p * ~mine).sum(1)
size = mine.sum(1)
lead = (op[:, b:] - oo[:, b:]).sum(1) / 24
c4 = c[:, :, b]
CAT = np.stack([c4 <= 4, c4 >= 5], -1).astype(np.float32)
amb = [(2, 5), (5, 9), (9, 17)]
W = np.zeros((N, len(keys), 2), np.float32)
for i in range(len(keys)):
    nmin, lb, thr, st, amt = PAR[i]
    low = np.ones((N, 8), bool)
    for k in range(int(st)): low &= c[:, :, b - k] <= thr
    w = np.floor(amt) * (low & mine[:, :, b] & (v[:, :, b] > 0)) * (size[:, b] >= nmin)[:, None]
    W[:, i] = np.einsum('nj,njk->nk', w.astype(np.float32), CAT)
P = W[np.arange(N), pid]
Rs, Es, labs = [], [], []
for k, cl in enumerate(['c<=4', 'c5+']):
    for lo, hi in amb:
        m = (PAR[:, 4] >= lo) & (PAR[:, 4] < hi)
        Pk = P[:, k] * m[pid]; Ek = (W[:, :, k] * m[None]).mean(1)
        Rs.append(Pk - Ek); Es.append(Ek); labs.append(f'{cl} amt{lo}-{hi}')
for yl, y in [('lead', lead), ('size r8', size[:, 8].astype(float)), ('own c rest', (c * mine)[:, :, 5:].sum(1).mean(1))]:
    X = np.column_stack([np.ones(N)] + Rs + Es)
    bb, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ bb
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (e ** 2).mean())
    print(yl, '  '.join(f'{l} {bb[i+1]:+.4f}({se[i+1]:.4f})' for i, l in enumerate(labs)))
