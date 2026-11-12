"""v5: points dealt per game at block starts (rounds 0, 4, 8, 12) split by context: target newcomer (not in my group
the round before) vs stayer, my group's size then (<=4 vs >=5), target contribution (<=2, 3-5, 6+). Then slot-FE
regression of lead (and own/other) over v2-v4 with these split features plus the other phases' totals."""
import numpy as np, sys
phases_other = {'r1-3': [1, 2, 3], 'r5-7': [5, 6, 7], 'r9-11': [9, 10, 11], 'r13-19': list(range(13, 20)), 'r20-23': [20, 21, 22, 23]}
split = sys.argv[1] if len(sys.argv) > 1 else 'new'
X, Y, S = [], {k: [] for k in ['lead', 'own', 'oth', 'mem']}, []
names = None
for v in ['v2', 'v3', 'v4']:
    zz = np.load(f'/tmp/pf/{v}.npz')
    c = zz['c']; p = zz['p'].astype(np.float32); m = zz['mine'].astype(bool); val = zz['v'].astype(np.float32); pt = zz['point']
    pool = 1.6 * c * val - p
    lead = ((pool * m).sum(1) - (pool * ~m).sum(1)).mean(1); own = (pool * m).sum(1).mean(1); oth = (pool * ~m).sum(1).mean(1); mem = m.sum(1).mean(1)
    feats = {}
    for r in [0, 4, 8, 12]:
        pr = p[:, :, r] * m[:, :, r]
        if split == 'new':
            if r == 0: groups = {'all': np.ones_like(pr, bool)}
            else: groups = {'new': ~m[:, :, r - 1], 'stay': m[:, :, r - 1]}
        elif split == 'size':
            n = m[:, :, r].sum(1, keepdims=True)
            groups = {'n<=4': np.broadcast_to(n <= 4, pr.shape), 'n>=5': np.broadcast_to(n >= 5, pr.shape)}
        elif split == 'c':
            cr = c[:, :, r]
            groups = {'c<=2': cr <= 2, 'c3-5': (cr >= 3) & (cr <= 5), 'c6+': cr >= 6}
        for g, msk in groups.items():
            feats[f'r{r}:{g}'] = (pr * msk).sum(1)
    for ph, rr in phases_other.items():
        feats[ph] = (p[:, :, rr] * m[:, :, rr]).sum((1, 2))
    names = list(feats)
    pts = np.unique(pt)
    Fm = np.zeros((len(pts), len(names)))
    for j, k in enumerate(pts):
        s = pt == k
        Fm[j] = [feats[n][s].mean() / 10 for n in names]
        for kk, a in zip(Y, [lead, own, oth, mem]): Y[kk].append(a[s].mean())
    X.append(Fm); S.append(pts)
    del c, p, m, val, pool, zz
X = np.concatenate(X); S = np.concatenate(S)
def demean(a):
    a = a.astype(float).copy()
    for s in np.unique(S):
        i = S == s; a[i] -= a[i].mean(0)
    return a
Xd = demean(X)
print('mean points/game per feature:', ' '.join(f'{n}:{x*10:.1f}' for n, x in zip(names, X.mean(0))))
for k in Y:
    y = demean(np.array(Y[k]))
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None); r = y - Xd @ b
    s2 = (r ** 2).sum() / (len(y) - 256 - len(names)); se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xd.T @ Xd)))
    print(k, ' '.join(f'{n}:{bb:+.2f}({ss:.2f})' for n, bb, ss in zip(names, b, se)))
