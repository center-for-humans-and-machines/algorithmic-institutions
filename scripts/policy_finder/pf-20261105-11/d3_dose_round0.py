"""v4: clean dose-response of round 1 contribution to round 0 punishment.
Round 0 histories are empty, so at fixed own c0 the own fine p0 varies only with the rule's params.
Also: the effect of the others' fines (vicarious), at fixed own c0 and p0."""
import numpy as np
cs, ps, vs = [], [], []
for tag in ['v1', 'v2', 'v3', 'ref']:
    z = np.load(f'/tmp/pf4/{tag}.npz', allow_pickle=True)
    cs.append(z['c']); ps.append(z['p']); vs.append(z['v'])
c = np.concatenate(cs).astype(float); p = np.concatenate(ps).astype(float); v = np.concatenate(vs)
ok = v[:, :, 0] & v[:, :, 1] & v.all(1)[:, 0][:, None]  # player valid r0,r1 and all valid in r0
c0, p0, c1 = c[:, :, 0], p[:, :, 0], c[:, :, 1]
oc0 = (c0.sum(1, keepdims=True) - c0) / 3
op0 = (p0.sum(1, keepdims=True) - p0) / 3
print('N', ok.sum())
pb = [0, 1, 3, 6, 10, 15, 20, 31]
for lo, hi in [(0, 3), (4, 7), (8, 10), (11, 13), (14, 16), (17, 19)]:
    m = ok & (c0 >= lo) & (c0 <= hi)
    row = []
    for a, b in zip(pb[:-1], pb[1:]):
        mm = m & (p0 >= a) & (p0 < b)
        row.append(f'p[{a},{b}):{c1[mm].mean():5.2f}(n{mm.sum():6d})' if mm.sum() > 200 else ' ' * 22)
    print(f'c0 {lo:2d}-{hi:2d}', ' '.join(row))
# linear regression c1 ~ c0 bins + own p0 spline + others p0 + others c0
m = ok
X = [np.ones(m.sum()), c0[m], oc0[m], np.minimum(p0[m], 3), np.clip(p0[m] - 3, 0, 7), np.clip(p0[m] - 10, 0, 30),
     (p0[m] > 0).astype(float), np.minimum(op0[m], 3), np.clip(op0[m] - 3, 0, 30)]
X = np.column_stack(X)
w, *_ = np.linalg.lstsq(X, c1[m], rcond=None)
print('coef: 1, c0, others c0, p0[0-3], p0[3-10], p0[10+], any p0, others p0 [0-3], [3+]')
print(np.round(w, 3))
