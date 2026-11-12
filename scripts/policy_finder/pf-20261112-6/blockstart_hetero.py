"""v3: heterogeneity of the block-start effect (v2 sweep natural experiment, see blockstart_effect.py), pooled over
b = 4, 8, 12, 16: does the rest-of-game lead per punishment point depend on the group's size or pool lead at b-1,
and on whom it hits (0 vs 1-3 vs 4-5 contributors; persistent or not)?"""
import numpy as np, json
D = np.load('/tmp/v2sweep.npz')
c, p, g, mg, name = D['c'], D['p'], D['g'], D['mine_g'], D['name']
pts = {q['name']: q['params'] for q in json.load(open('plots/simulation/policy_finder/pf-20261112-6-v2_sweep/sweep.json'))['points']}
nb = np.array([pts[n]['n_blocks'] if pts[n]['amt'] >= 1 else 0 for n in name])
mine = g == mg[:, None, None]
own_pool = (1.6 * c * mine - p * mine).sum(1); oth_pool = (1.6 * c * ~mine - p * ~mine).sum(1)
size = mine.sum(1).astype(float)
own_c = (c * mine).sum(1); oth_c = (c * ~mine).sum(1)
Xs, Ys, Ss, Ls = [], [], [], []
for k in [1, 2, 3, 4]:
    b = 4 * k
    sel = (nb == k) | (nb == k + 1)
    Pb = (p[:, :, b] * mine[:, :, b]).sum(1)
    lowb = ((c[:, :, b] <= 3) * mine[:, :, b]).sum(1)
    big = (size[:, b] >= 5).astype(float)
    ahead = ((own_pool[:, b - 1] - oth_pool[:, b - 1]) > 0).astype(float)
    ctrl = [own_c[:, b], oth_c[:, b], size[:, b], lowb, mg, own_c[:, b - 1], (p[:, :, :b] * mine[:, :, :b]).sum((1, 2)), big, ahead]
    dummies = [np.full(len(c), float(k == kk)) for kk in [2, 3, 4]]
    X = np.column_stack([np.ones(len(c)), Pb * big, Pb * (1 - big), Pb * ahead] + ctrl + dummies)[sel]
    Xs.append(X); Ys.append(((own_pool[:, b:] - oth_pool[:, b:]).sum(1) / 24)[sel]); Ss.append(size[sel, min(b + 4, 23)])
X = np.concatenate(Xs)
for lab, y in [('lead_rest', np.concatenate(Ys)), ('size b+4', np.concatenate(Ss))]:
    XtXi = np.linalg.inv(X.T @ X); bb = XtXi @ X.T @ y; r = y - X @ bb; se = np.sqrt(np.diag(XtXi) * (r ** 2).mean())
    print(lab, 'P*big %+.4f(%.4f)  P*small %+.4f(%.4f)  P*ahead(extra) %+.4f(%.4f)' % (bb[1], se[1], bb[2], se[2], bb[3], se[3]))
