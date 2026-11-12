"""v3: within the v2 block-start experiment (blockstart_iv.py), games whose group had >= 7 members (and, separately,
>= 6) at b: split the per-point effect on the rest-of-game lead by the point's thr (0-3 vs 4-8) and amt (< 4 vs >= 4),
and by block starts. Also how often each size occurs at each block start under zero."""
import numpy as np, json
D = np.load('/tmp/v2sweep.npz')
c, p, g, mg, name, v = D['c'], D['p'], D['g'], D['mine_g'], D['name'], D['v']
pts = {q['name']: q['params'] for q in json.load(open('plots/simulation/policy_finder/pf-20261112-6-v2_sweep/sweep.json'))['points']}
par = lambda k: np.array([pts[n][k] for n in name], dtype=float)
nb, thr, streak, amt = par('n_blocks'), par('thr'), par('streak'), par('amt')
nb = np.where(amt >= 1, nb, 0)
mine = g == mg[:, None, None]
own_pool = (1.6 * c * mine - p * mine).sum(1); oth_pool = (1.6 * c * ~mine - p * ~mine).sum(1)
size = mine.sum(1).astype(float); own_c = (c * mine).sum(1); oth_c = (c * ~mine).sum(1)
z = nb == 0
print('own size distribution under zero-equivalent points:')
for t in [4, 8, 12, 16, 20]:
    print(f'  round {t}:', ' '.join(f'{L}:{(size[z, t] == L).mean():.2f}' for L in range(9)))
def run(nmin, splits, bs):
    Xs, ys = [], []
    for b in bs:
        k = b // 4
        sel = ((nb == k) | (nb == k + 1)) & (amt >= 1) & (size[:, b] >= nmin)
        low = np.ones(c.shape[:2], bool)
        for s in range(3):
            low &= np.where(streak[:, None] > s, c[:, :, b - s] <= thr[:, None], True)
        Pw = (np.floor(amt)[:, None] * low * mine[:, :, b] * (v[:, :, b] > 0)).sum(1)
        T = (nb == k + 1).astype(float)
        cols = [np.ones(len(c))]
        for f in splits.values(): cols.append(T * Pw * f)
        for f in splits.values(): cols += [Pw * f, f, T * f]
        cols += [own_c[:, b], oth_c[:, b], mg, own_c[:, b - 1], (p[:, :, :b] * mine[:, :, :b]).sum((1, 2)), size[:, b]]
        cols += [np.full(len(c), float(b == bb)) for bb in bs[1:]]
        Xs.append(np.column_stack(cols)[sel]); ys.append(((own_pool[:, b:] - oth_pool[:, b:]).sum(1) / 24)[sel])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    XtXi = np.linalg.pinv(X.T @ X); bb = XtXi @ X.T @ y; r = y - X @ bb; se = np.sqrt(np.abs(np.diag(XtXi)) * (r ** 2).mean())
    return '  '.join(f'{k}:{bb[1 + i]:+.3f}({se[1 + i]:.3f})' for i, k in enumerate(splits))
for nmin in [7, 6]:
    for bs in [[4, 8], [4, 8, 12, 16]]:
        print(f'n>={nmin} b={bs}')
        print('   thr  ', run(nmin, {'thr0-3': (thr <= 3).astype(float), 'thr4-8': (thr >= 4).astype(float)}, bs))
        print('   amt  ', run(nmin, {'amt<4': (amt < 4).astype(float), 'amt>=4': (amt >= 4).astype(float)}, bs))
        print('   strk ', run(nmin, {'streak1': (streak == 1).astype(float), 'streak2-3': (streak >= 2).astype(float)}, bs))
