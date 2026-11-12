"""Across v1 sweep points: regress the lead (and own size, own pool) on where the punishment went
(per game-round, punishment points dealt to members by their contribution that round, and by phase)."""
import numpy as np
z = np.load('/tmp/v1_compact.npz'); c, p, m, pt = [z[k].astype(np.float32) for k in ['c', 'p', 'mine', 'point']]
pt = pt.astype(int)
own = (1.6 * c * m - p * m).sum(1).mean(-1); oth = (1.6 * c * (1 - m) - p * (1 - m)).sum(1).mean(-1)
lead = own - oth; size = m.sum(1).mean(-1)
pm = p * m
feats = {}
for lo, hi in [(0, 2), (3, 6), (7, 11), (12, 20)]:
    feats[f'p_c{lo}-{hi}'] = (pm * ((c >= lo) & (c <= hi))).sum((1, 2)) / 24
ph = np.zeros(24, int); ph[20:] = 1
feats['p_r20+'] = pm[:, :, 20:].sum((1, 2)) / 24
names = list(feats)
F = np.stack([np.bincount(pt, f) / np.bincount(pt) for f in feats.values()], 1)
Ys = {k: np.bincount(pt, y) / np.bincount(pt) for k, y in [('lead', lead), ('size', size), ('own', own), ('oth', oth)]}
print('mean feature values', dict(zip(names, F.mean(0).round(2))))
for k, y in Ys.items():
    # fit on points with modest punishment to avoid the collapse region dominating
    for cap in [3, 30]:
        s = F[:, :4].sum(1) < cap
        A = np.c_[np.ones(s.sum()), F[s]]
        b, res, *_ = np.linalg.lstsq(A, y[s], rcond=None)
        r = y[s] - A @ b
        cov = np.linalg.inv(A.T @ A) * r.var()
        print(k, f'cap{cap} n={s.sum()}', ' '.join(f'{n}:{bb:.2f}({se:.2f})' for n, bb, se in zip(['icpt'] + names, b, np.sqrt(np.diag(cov)))))
