"""Pooled sweeps (every 4th episode): does the exit cost of own punishment depend on how my group's
per-capita common good compares with the other group's? Leave rate at decisions 3, 7, 11 for my
members, by own phase punishment bin x per-capita gap (own minus other, phase means, gap computed
without this member's own punishment)."""
import numpy as np
vers = ['pf-20261104-9', 'pf-20261104-9-v2', 'pf-20261104-9-v3', 'pf-20261104-9-v4']
A = []
for v in vers:
    z = np.load(f'/tmp/pf/{v}.npz'); C = z['C'][:, ::4].astype(np.float32); V = z['V'][:, ::4]; P = z['P'][:, ::4].astype(np.float32); G = z['G'][:, ::4]
    cg = []
    for g in (0, 1):
        m = (G == g) & (V == 1)
        cg.append((1.6 * (C * m).sum(2) - (P * m).sum(2)) / np.maximum(m.sum(2), 1))
    for d in (3, 7, 11):
        rs = slice(d - 3, d + 1)
        sel = (G[..., d] == 0) & (V[..., d] == 1)
        gap = (cg[0][..., rs].mean(-1) - cg[1][..., rs].mean(-1))[..., None].repeat(8, -1)[sel]
        n = (G[..., d] == 0).sum(-1)[..., None].repeat(8, -1)[sel]
        pown = P[..., rs].sum(-1)[sel]
        gap = gap + pown / 4 / n   # remove own punishment's share from the gap
        A.append(np.column_stack([np.full(sel.sum(), d), (G[..., d + 1] != 0)[sel], gap, pown]))
A = np.concatenate(A); d, leave, gap, pown = A.T
print('n', len(A))
gb = [(-99, -4), (-4, 0), (0, 4), (4, 99)]
pb = [(0, 1), (1, 16), (16, 41), (41, 999)]
print('rows: gap bins; cols: own punishment over phase 0 / 1-15 / 16-40 / 41+')
for glo, ghi in gb:
    line = f'gap {glo:4d}..{ghi:3d}:'
    for plo, phi in pb:
        s = (gap >= glo) & (gap < ghi) & (pown >= plo) & (pown < phi)
        line += f'  {leave[s].mean():.3f} (n{s.sum()//1000}k)'
    print(line)
