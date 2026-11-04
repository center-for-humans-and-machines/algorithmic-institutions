"""Heterogeneous treatment effects of the rule's dials on the pool, using the
Sobol randomisation of the sweeps: `decay` (and `size`) act only from round 4,
so the state at round 4 is pre-treatment for them; `slope` acts from round 0, so
round-0 contributions are pre-treatment for it. OLS with point fixed effects:
Y = FE_point + b*X + sum_k g_k * X * param_k ; we report g for the treatment.
usage: v6_hetero.py v4 v5"""
import numpy as np, sys
tags = sys.argv[1:]
rows = []
for tag in tags:
    z = np.load(f'/tmp/{tag}.npz')
    c, p, g = z['c'].astype(np.float32), z['p'].astype(np.float32), z['g']
    keys = list(z['keys']); par = z['params']
    if 'size' not in keys:
        par = np.c_[par, np.zeros(len(par))]; keys = keys + ['size']
    P, E = c.shape[:2]
    mine = (g == 0).astype(np.float32)
    pool = (1.6 * c - p) * mine                   # [P,E,8,24]
    pr = pool.sum(2)                               # [P,E,24]
    n = mine.sum(2)                                # [P,E,24]
    oth = 1 - mine
    def gmean(x, m, t):
        return (x[..., t] * m[..., t]).sum(-1) / np.maximum(m[..., t].sum(-1), 1)
    X = dict(
        n4=n[..., 4],
        own3=gmean(c, mine, 3), riv3=gmean(c, oth, 3),
        own_new4=gmean(c, mine, 4), riv4=gmean(c, oth, 4),
        own0=gmean(c, mine, 0), riv0=gmean(c, oth, 0),
        low0=((c[..., 0] <= 4) * mine[..., 0]).sum(-1),
    )
    Y_late = pr[..., 4:].mean(-1); Y_all = pr.mean(-1)
    for i in range(P):
        for e in range(E):
            rows.append([i + 1000 * tags.index(tag), *par[i], *[X[k][i, e] for k in X], Y_late[i, e], Y_all[i, e]])
xk = ['n4', 'own3', 'riv3', 'own_new4', 'riv4', 'own0', 'riv0', 'low0']
pk = ['thr', 'peer', 'slope', 'decay', 'follow', 'size']
R = np.array(rows)
pt = R[:, 0].astype(int); prm = R[:, 1:7].copy(); Xs = R[:, 7:15]; yl = R[:, 15]; ya = R[:, 16]
prm[:, 2] = np.log(prm[:, 2]); prm[:, 3] = np.log(prm[:, 3])   # log slope, log decay
prm = (prm - prm.mean(0)) / prm.std(0)
_, pti = np.unique(pt, return_inverse=True)
def demean(a):
    s = np.zeros((pti.max() + 1,) + a.shape[1:]); np.add.at(s, pti, a)
    cnt = np.bincount(pti).reshape(-1, *([1] * (a.ndim - 1)))
    return a - (s / cnt)[pti]
def fit(y, cov_idx, label):
    cols, names = [], []
    for j in cov_idx:
        x = Xs[:, j]; xs = (x - x.mean()) / x.std()
        cols.append(xs); names.append(xk[j])
        for q in range(6):
            cols.append(xs * prm[:, q]); names.append(f'{xk[j]}*{pk[q]}')
    A = demean(np.column_stack(cols)); yy = demean(y[:, None])[:, 0]
    b, *_ = np.linalg.lstsq(A, yy, rcond=None)
    r = yy - A @ b
    cov = np.linalg.inv(A.T @ A) * (r @ r) / (len(yy) - A.shape[1] - pti.max())
    se = np.sqrt(np.diag(cov))
    print(f'--- {label}: Y sd {y.std():.1f}; coefficients per 1 sd of X (and per 1 sd of the param)')
    for nm, bb, ss in zip(names, b, se):
        flag = ' *' if abs(bb) > 3 * ss else ''
        print(f'  {nm:22s} {bb:7.2f} ({ss:.2f}){flag}')
print('mean pool late', yl.mean(), 'all', ya.mean())
fit(yl, [0, 1, 2], 'pool rounds 4-23 on state at round 3/4 (decay, size pre-treated)')
fit(ya, [5, 6, 7], 'pool rounds 0-23 on round-0 contributions')
