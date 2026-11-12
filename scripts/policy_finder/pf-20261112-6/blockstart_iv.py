"""v3: cleaner block-start experiment. For every game of v2 points with n_blocks in {k, k+1} I rebuild the punishment
the rule *would* deal at b = 4k (floor(amt) to each valid member with c <= thr in the last `streak` rounds), Pw.
Treatment T = (n_blocks == k+1) is random given the point's other params; actual punishment = T * Pw. Regress
outcomes on T*Pw by own group size at b, controlling for Pw (by size) and the state, so only T varies."""
import numpy as np, json
D = np.load('/tmp/v2sweep.npz')
c, p, g, mg, name, v = D['c'], D['p'], D['g'], D['mine_g'], D['name'], D['v']
pts = {q['name']: q['params'] for q in json.load(open('plots/simulation/policy_finder/pf-20261112-6-v2_sweep/sweep.json'))['points']}
par = lambda k: np.array([pts[n][k] for n in name], dtype=float)
nb, thr, streak, amt = par('n_blocks'), par('thr'), par('streak'), par('amt')
nb = np.where(amt >= 1, nb, 0)
mine = g == mg[:, None, None]
own_pool = (1.6 * c * mine - p * mine).sum(1); oth_pool = (1.6 * c * ~mine - p * ~mine).sum(1)
size = mine.sum(1).astype(float)
own_c = (c * mine).sum(1); oth_c = (c * ~mine).sum(1)
groups = {'n<=4': lambda s: s <= 4, 'n=5': lambda s: s == 5, 'n=6': lambda s: s == 6, 'n>=7': lambda s: s >= 7}
for bs in [[4, 8], [12, 16], [4, 8, 12, 16]]:
    Xs, Y = [], {'lead_rest': [], 'size_b+4': [], 'ownc_rest_per_round': [], 'othc_rest_per_round': []}
    cnt = {k: 0 for k in groups}
    for b in bs:
        k = b // 4
        sel = ((nb == k) | (nb == k + 1)) & (amt >= 1)
        low = np.ones(c.shape[:2], bool)
        for s in range(3):
            low &= np.where(streak[:, None] > s, c[:, :, b - s] <= thr[:, None], True)
        Pw = (np.floor(amt)[:, None] * low * mine[:, :, b] * (v[:, :, b] > 0)).sum(1)
        T = (nb == k + 1).astype(float)
        assert np.allclose((p[:, :, b] * mine[:, :, b]).sum(1)[sel], (T * Pw)[sel]), 'rebuild mismatch'
        sz = size[:, b]
        cols = [np.ones(len(c))]
        for kk, f in groups.items():
            m = f(sz).astype(float)
            cols += [T * Pw * m]
            cnt[kk] += int((sel & (T * Pw * m > 0)).sum())
        for kk, f in groups.items():
            m = f(sz).astype(float)
            cols += [Pw * m, m, T * m]
        cols += [own_c[:, b], oth_c[:, b], mg, own_c[:, b - 1], (p[:, :, :b] * mine[:, :, :b]).sum((1, 2)), sz]
        cols += [np.full(len(c), float(b == bb)) for bb in bs[1:]]
        Xs.append(np.column_stack(cols)[sel])
        Y['lead_rest'].append(((own_pool[:, b:] - oth_pool[:, b:]).sum(1) / 24)[sel])
        Y['size_b+4'].append(size[sel, min(b + 4, 23)])
        Y['ownc_rest_per_round'].append(own_c[sel, b + 1:].mean(1))
        Y['othc_rest_per_round'].append(oth_c[sel, b + 1:].mean(1))
    X = np.concatenate(Xs)
    print('block starts', bs, 'treated games', cnt)
    for lab, y in Y.items():
        y = np.concatenate(y)
        XtXi = np.linalg.pinv(X.T @ X); bb = XtXi @ X.T @ y; r = y - X @ bb; se = np.sqrt(np.abs(np.diag(XtXi)) * (r ** 2).mean())
        print(f'   {lab:22s}', '  '.join(f'{kk}:{bb[1 + i]:+.4f}({se[1 + i]:.4f})' for i, kk in enumerate(groups)))
