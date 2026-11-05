# v5: what drives leaving (members of a group) and joining (members of the other group) at the
# decisions after rounds 3, 7, 11 in v4's sweep: logistic regression on own punishment and the
# per-head pool of both groups in each round of the block, own contribution relative to group.
import numpy as np, statsmodels.api as sm
d = np.load('/tmp/v4sweep.npz')
c, p, v, g = d['c'].astype(float), d['p'].astype(float), d['v'].astype(bool), d['g']
N = len(c)
rng = np.random.default_rng(0)
idx = rng.choice(N, 40000, replace=False)
c, p, v, g = c[idx], p[idx], v[idx], g[idx]
def cgph(grp):  # per-head pool of group grp per round [N, 24]
    m = (g == grp) & v
    n = m.sum(1)
    return np.where(n > 0, (1.6 * (c * m).sum(1) - (p * m).sum(1)) / np.maximum(n, 1), 0), (g == grp).sum(1)
cg0, n0 = cgph(0); cg1, n1 = cgph(1)
for dec in (3, 7, 11):
    rows, ys = [], []
    for i in range(8):
        gi = g[:, i, dec]
        stay = (g[:, i, dec - 3:dec + 1] == gi[:, None]).all(1)
        own = np.where(gi[:, None] == 0, cg0, cg1)[:, dec - 3:dec + 1]
        oth = np.where(gi[:, None] == 0, cg1, cg0)[:, dec - 3:dec + 1]
        nown = np.where(gi == 0, n0[:, dec], n1[:, dec])
        X = np.column_stack([p[:, i, dec - 3:dec + 1], own / 20, oth / 20, c[:, i, dec] / 20, nown])
        rows.append(X[stay]); ys.append((g[:, i, dec + 1] != gi)[stay])
    X = np.concatenate(rows); y = np.concatenate(ys).astype(float)
    names = [f'p_r{k}' for k in range(4)] + [f'cgown_r{k}' for k in range(4)] + [f'cgoth_r{k}' for k in range(4)] + ['c_dec', 'n_own']
    r = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    print(f'decision after round {dec}: n={len(y)} leave rate {y.mean():.3f}')
    for nm, b, se in zip(['const'] + names, r.params, r.bse):
        print(f'  {nm:10s} {b:+.3f} ({se:.3f})')
