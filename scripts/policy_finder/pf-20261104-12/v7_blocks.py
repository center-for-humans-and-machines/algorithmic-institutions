"""Per-block value of punishing 0-9 givers: per point, conditional mean dose to givers of 0-9
at each block's positions 0-1 (summed), and to 10+ givers at block 0; pool regressed on them,
with version dummies, per version and pooled. Also the members and the bill per block."""
import numpy as np
F_, Y_, V_ = [], [], []
for v in ['v3', 'v4', 'v5', 'v6']:
    z = np.load(f'/tmp/pf-20261104-12-{v}.npz')
    c, p, g = z['c'], z['p'].astype(np.float32), z['g']
    our = g == 0
    f = []
    for b in range(5):
        rs = [4*b, 4*b+1]
        m = our[..., rs] & (c[..., rs] <= 9)
        f.append(sum(((p[..., [r]] * m[..., [rs.index(r)]]).sum((1, 2, 3)) / np.maximum(m[..., [rs.index(r)]].sum((1, 2, 3)), 1)) for r in rs))
    m = our[..., [0, 1]] & (c[..., [0, 1]] >= 10)
    f.append((p[..., [0, 1]] * m).sum((1, 2, 3)) / np.maximum(m.sum((1, 2, 3)), 1) * 2)
    F_.append(np.stack(f, 1)); Y_.append(z['pool']); V_ += [v] * 256
F = np.concatenate(F_); Y = np.concatenate(Y_); V = np.array(V_)
hn = ['b0', 'b1', 'b2', 'b3', 'b4', 'b0_10+']
print('mean doses', np.round(F.mean(0), 1), 'sd', np.round(F.std(0), 1))
print(np.round(np.corrcoef(F.T), 2))
for v in ['all', 'v3', 'v4', 'v5', 'v6']:
    s = np.ones(len(Y), bool) if v == 'all' else V == v
    Zv = np.column_stack([np.ones(s.sum()), F[s]] + ([(V[s] == w).astype(float) for w in ['v4', 'v5', 'v6']] if v == 'all' else []))
    bv = np.linalg.lstsq(Zv, Y[s], rcond=None)[0]; rv = Y[s] - Zv @ bv
    sev = np.sqrt(np.diag(np.linalg.inv(Zv.T @ Zv)) * rv.var())
    print(v, ' '.join(f'{n}:{bv[1+i]:+.3f}({sev[1+i]:.3f})' for i, n in enumerate(hn)))
# bins: block-2 dose high vs low, block-3/4 dose high vs low, among points with b0 and b1 doses in the top half
hi01 = (F[:, 0] > np.median(F[:, 0])) & (F[:, 1] > np.median(F[:, 1]))
late = F[:, 3] + F[:, 4]
for lo2, hi2 in [(0, 15), (15, 30), (30, 45), (45, 61)]:
    for lo3, hi3 in [(0, 10), (10, 30), (30, 130)]:
        s = hi01 & (F[:, 2] >= lo2) & (F[:, 2] < hi2) & (late >= lo3) & (late < hi3)
        if s.sum() >= 5: print(f'b2 {lo2}-{hi2} b34 {lo3}-{hi3}: n={s.sum():3d} pool {Y[s].mean():.2f} ({Y[s].std()/np.sqrt(s.sum()):.2f})')
np.savez('/tmp/v7_blocks.npz', F=F, Y=Y, V=V)
