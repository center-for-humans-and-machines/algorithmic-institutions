"""v6: in the clone sweeps' early rounds (1-6), what do near-full contributors (c 14-18)
do 1 and 3 rounds after getting 0 / 1 / 2 / 3 / 4-6 points, by own c and the others' mean?"""
import sys, numpy as np
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
import v3_load

rows = []
for ver in ['pf-20261105-3-v2', 'pf-20261105-3-v3', 'pf-20261105-3-v4', 'pf-20261105-3-v5']:
    a = v3_load.load(ver)
    c, p, v = a['c'], a['p'], a['v']
    c = np.where(v, c, np.nan)
    for t in range(1, 7):
        ct = c[..., t]; pt = p[..., t]
        oth = (np.nansum(c[..., t], -1, keepdims=True) - np.nan_to_num(ct)) / np.maximum(v[..., t].sum(-1, keepdims=True) - v[..., t], 1)
        d1 = c[..., t + 1] - ct; d3 = c[..., t + 3] - ct
        # others' change 3 rounds later
        o3 = (np.nansum(c[..., t + 3], -1, keepdims=True) - np.nan_to_num(c[..., t + 3])) / 3 - oth
        rows.append(np.stack([ct, pt, oth, d1, d3, o3, np.full_like(ct, t)], -1).reshape(-1, 7))
R = np.concatenate(rows); R = R[~np.isnan(R).any(1)]
print('rows', len(R))
for cl in [(14, 15), (16, 17), (18, 19)]:
    for ob in [(10, 14), (14, 17), (17, 21)]:
        s = (R[:, 0] >= cl[0]) & (R[:, 0] <= cl[1]) & (R[:, 2] >= ob[0]) & (R[:, 2] < ob[1])
        out = []
        for pb in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 6)]:
            q = s & (R[:, 1] >= pb[0]) & (R[:, 1] <= pb[1])
            if q.sum() > 200:
                out.append(f'p{pb[0]}-{pb[1]}: n{q.sum()} d1 {R[q,3].mean():+.2f} d3 {R[q,4].mean():+.2f} oth3 {R[q,5].mean():+.2f}')
        print(f'c {cl} others {ob}:', ' | '.join(out))
