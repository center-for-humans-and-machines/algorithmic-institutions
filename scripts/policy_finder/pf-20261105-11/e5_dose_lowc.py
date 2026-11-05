"""v5: round-0 dose response of low contributors, finer fine bins up to the cap; c1, c2, c3 and own fines p1+p2.
Also the others' mean c1-c3 in the same games (the fine on the low player is part of the others' vicarious dose)."""
import numpy as np
cs, ps, vs = [], [], []
for tag in ['v1', 'v2', 'v3', 'v4']:
    z = np.load(f'/tmp/pf5/{tag}.npz', allow_pickle=True)
    cs.append(z['c'][:, :, :6]); ps.append(z['p'][:, :, :6]); vs.append(z['v'][:, :, :6])
c = np.concatenate(cs).astype(float); p = np.concatenate(ps).astype(float); v = np.concatenate(vs)
ok = v[:, :, :4].all((1, 2))
for lo, hi in [(0, 2), (3, 5), (6, 8)]:
    print(f'own c0 {lo}-{hi}')
    for a, b in [(10, 15), (15, 18), (18, 21), (21, 24), (24, 27), (27, 30), (30, 31)]:
        out = []
        for i in range(4):
            m = ok & (c[:, i, 0] >= lo) & (c[:, i, 0] <= hi) & (p[:, i, 0] >= a) & (p[:, i, 0] < b)
            out.append(np.column_stack([c[m, i, 1], c[m, i, 2], c[m, i, 3], p[m, i, 1] + p[m, i, 2], np.delete(c[m], i, 1)[:, :, 1:4].mean((1, 2))]))
        O = np.concatenate(out)
        if len(O) > 300: print(f'  p0 [{a},{b}) n{len(O):6d} c1 {O[:,0].mean():5.2f} c2 {O[:,1].mean():5.2f} c3 {O[:,2].mean():5.2f} own p1+p2 {O[:,3].mean():5.2f} others c1-3 {O[:,4].mean():5.2f}')
