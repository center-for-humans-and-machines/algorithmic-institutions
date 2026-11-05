"""v5: vicarious response at round 0. Players with own p0 = 0, at fixed own c0 bin and fixed others' c0 (sorted, binned):
does c1 rise with the others' total fine p0? (others' fines vary only with the rule's params here)."""
import numpy as np
cs, ps, vs = [], [], []
for tag in ['v1', 'v2', 'v3', 'v4', 'ref']:
    z = np.load(f'/tmp/pf5/{tag}.npz', allow_pickle=True)
    cs.append(z['c'][:, :, :3]); ps.append(z['p'][:, :, :3]); vs.append(z['v'][:, :, :3])
c = np.concatenate(cs).astype(float); p = np.concatenate(ps).astype(float); v = np.concatenate(vs)
allv = v[:, :, 0].all(1) & v[:, :, 1].all(1)
rows = []
for i in range(4):
    o = [j for j in range(4) if j != i]
    oc = np.sort(c[:, o, 0], 1)
    rows.append(np.column_stack([c[:, i, 0], p[:, i, 0], c[:, i, 1], c[:, i, 2], oc, p[:, o, 0].sum(1), c[:, o, 1].mean(1), allv]))
R = np.concatenate(rows)
R = R[R[:, -1] == 1]
c0, p0, c1, c2, o1, o2, o3, op, oc1, _ = R.T
print('N', len(R))
for lo, hi in [(10, 13), (14, 16), (17, 20)]:
    m0 = (c0 >= lo) & (c0 <= hi) & (p0 == 0)
    print(f'own c0 {lo}-{hi}, own p0=0')
    # condition on others' c0 profile cells: min bin x mid bin
    for mlo, mhi in [(0, 4), (5, 9)]:
        for dlo, dhi in [(5, 11), (12, 16)]:
            m = m0 & (o1 >= mlo) & (o1 <= mhi) & (o2 >= dlo) & (o2 <= dhi)
            out = []
            for a, b in [(0, 10), (10, 20), (20, 30), (30, 45), (45, 91)]:
                mm = m & (op >= a) & (op < b)
                if mm.sum() > 300: out.append(f'op[{a},{b}) c1 {c1[mm].mean():5.2f} c2 {c2[mm].mean():5.2f} oc1 {oc1[mm].mean():5.2f} n{mm.sum()}')
            print(f'  others min {mlo}-{mhi} mid {dlo}-{dhi}:', ' | '.join(out))
