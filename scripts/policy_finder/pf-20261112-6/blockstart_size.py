"""v3: the block-start natural experiment (v2 sweep, b = 4/8/12/16 pooled) split by own group size at b:
rest-of-game lead and size at b+4 per punishment point, one coefficient per size level, plus the same split for the
low-contributor share. Also per-member contributions after."""
import numpy as np, json
D = np.load('/tmp/v2sweep.npz')
c, p, g, mg, name = D['c'], D['p'], D['g'], D['mine_g'], D['name']
pts = {q['name']: q['params'] for q in json.load(open('plots/simulation/policy_finder/pf-20261112-6-v2_sweep/sweep.json'))['points']}
nb = np.array([pts[n]['n_blocks'] if pts[n]['amt'] >= 1 else 0 for n in name])
mine = g == mg[:, None, None]
own_pool = (1.6 * c * mine - p * mine).sum(1); oth_pool = (1.6 * c * ~mine - p * ~mine).sum(1)
size = mine.sum(1).astype(float)
own_c = (c * mine).sum(1); oth_c = (c * ~mine).sum(1)
levels = [1, 2, 3, 4, 5, 6, 7, 8]
for bs in [[4, 8, 12, 16], [4, 8], [12, 16]]:
    Xs, Ys, Ss, Ns = [], [], [], []
    for b in bs:
        k = b // 4
        sel = (nb == k) | (nb == k + 1)
        Pb = (p[:, :, b] * mine[:, :, b]).sum(1)
        lowb = ((c[:, :, b] <= 3) * mine[:, :, b]).sum(1)
        sz = size[:, b]
        inter = [Pb * (sz == L) for L in levels]
        sdum = [(sz == L).astype(float) for L in levels[1:]]
        ctrl = [own_c[:, b], oth_c[:, b], lowb, mg, own_c[:, b - 1], (p[:, :, :b] * mine[:, :, :b]).sum((1, 2))]
        dummies = [np.full(len(c), float(b == bb)) for bb in bs[1:]]
        X = np.column_stack([np.ones(len(c))] + inter + sdum + ctrl + dummies)[sel]
        Xs.append(X); Ys.append(((own_pool[:, b:] - oth_pool[:, b:]).sum(1) / 24)[sel]); Ss.append(size[sel, min(b + 4, 23)])
        Ns.append(np.column_stack([sz, Pb > 0])[sel])
    X = np.concatenate(Xs); N = np.concatenate(Ns)
    print('block starts', bs)
    for lab, y in [('lead_rest', np.concatenate(Ys)), ('size b+4', np.concatenate(Ss))]:
        XtXi = np.linalg.pinv(X.T @ X); bb = XtXi @ X.T @ y; r = y - X @ bb; se = np.sqrt(np.abs(np.diag(XtXi)) * (r ** 2).mean())
        print('  ', lab, '  '.join(f'n={L}:{bb[1 + i]:+.3f}({se[1 + i]:.3f})[{int(((N[:, 0] == L) & (N[:, 1] > 0)).sum())}]' for i, L in enumerate(levels)))
