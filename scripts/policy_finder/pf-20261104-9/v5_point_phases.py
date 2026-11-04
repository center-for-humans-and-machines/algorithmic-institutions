"""Per design point of all four sweeps: pool by phase (rounds 0-3, 4-11, 12-23), members, contribution
per member and punishment per member late; then the frontier between early cost and late plateau."""
import numpy as np, json
vers = ['pf-20261104-9', 'pf-20261104-9-v2', 'pf-20261104-9-v3', 'pf-20261104-9-v4']
out = []
for vi, v in enumerate(vers):
    z = np.load(f'/tmp/pf/{v}.npz'); C = z['C'].astype(np.float32); V = z['V']; P = z['P'].astype(np.float32); G = z['G']
    m = (G == 0) & (V == 1)
    pool = (1.6 * C * m - P * m).sum(2)          # [pt, ep, r]
    n = (G == 0).sum(2)
    cm = (C * m).sum(2) / np.maximum(m.sum(2), 1)
    pm = (P * m).sum(2) / np.maximum(m.sum(2), 1)
    # other group
    m1 = (G == 1) & (V == 1)
    pool1 = (1.6 * C * m1 - P * m1).sum(2)
    cm1 = (C * m1).sum(2) / np.maximum(m1.sum(2), 1)
    for pt in range(C.shape[0]):
        f = lambda a, rs: a[pt][:, rs].mean()
        out.append([vi, pt, f(pool, slice(0, 24)), f(pool, slice(0, 4)), f(pool, slice(4, 12)), f(pool, slice(12, 24)),
                    f(n, slice(4, 12)), f(n, slice(12, 24)), f(cm, slice(12, 24)), f(pm, slice(0, 4)), f(pm, slice(4, 12)),
                    f(pm, slice(12, 24)), f(cm1, slice(12, 24)), f(pool1, slice(0, 24)), f(cm, slice(20, 24))])
O = np.array(out)
np.save('/tmp/pf/point_phases.npy', O)
cols = 'ver pt pool p0_3 p4_11 p12_23 n4_11 n12_23 c12_23 pun0_3 pun4_11 pun12_23 cAH12_23 poolAH c20_23'.split()
print(' '.join(f'{c:>8s}' for c in cols))
for row in O[np.argsort(-O[:, 2])][:15]:
    print(' '.join(f'{x:8.2f}' for x in row))
print('--- top by late pool 12-23')
for row in O[np.argsort(-O[:, 5])][:10]:
    print(' '.join(f'{x:8.2f}' for x in row))
print('--- top by late c per member')
for row in O[np.argsort(-O[:, 8])][:10]:
    print(' '.join(f'{x:8.2f}' for x in row))
print('corr of total pool with:', {c: round(np.corrcoef(O[:, 2], O[:, i])[0, 1], 2) for i, c in enumerate(cols) if i > 2})
