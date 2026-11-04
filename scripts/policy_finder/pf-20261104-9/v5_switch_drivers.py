"""Pooled sweeps: linear-probability models of (a) my members leaving and (b) ah's members joining at
each decision, on own punishment by phase position, per-capita common good of both groups (mean over
the phase and in the pre-decision round), own c / others' mean, group sizes, decision dummies."""
import numpy as np
vers = ['pf-20261104-9', 'pf-20261104-9-v2', 'pf-20261104-9-v3', 'pf-20261104-9-v4']
LA, JA = [], []
for v in vers:
    z = np.load(f'/tmp/pf/{v}.npz'); C = z['C'][::2].astype(np.float32); V = z['V'][::2]; P = z['P'][::2].astype(np.float32); G = z['G'][::2]
    cg = []
    for g in (0, 1):
        m = (G == g) & (V == 1)
        cg.append((1.6 * (C * m).sum(2) - (P * m).sum(2)) / np.maximum(m.sum(2), 1))   # [pt, ep, r]
    nsz = [(G == g).sum(2) for g in (0, 1)]
    for d in [3, 7, 11, 15, 19]:
        rs = list(range(d - 3, d + 1))
        for g, out in ((0, LA), (1, JA)):
            sel = G[..., d] == g
            leave = (G[..., d + 1] != g)[sel]
            own = cg[g][..., rs].mean(-1)[..., None].repeat(8, -1)[sel]
            other = cg[1 - g][..., rs].mean(-1)[..., None].repeat(8, -1)[sel]
            own3 = cg[g][..., d][..., None].repeat(8, -1)[sel]
            oth3 = cg[1 - g][..., d][..., None].repeat(8, -1)[sel]
            pp = [P[..., r][sel] for r in rs]
            m = (G[..., d] == g) & (V[..., d] == 1)
            k = m.sum(-1, keepdims=True)
            o = ((C[..., d] * m).sum(-1, keepdims=True) - C[..., d] * m) / np.maximum(k - m, 1)
            rel = (C[..., d] / np.maximum(o, 0.5))[sel]
            n = nsz[g][..., d][..., None].repeat(8, -1)[sel]
            out.append(np.column_stack([np.full(leave.shape, d), leave, own, other, own3, oth3] + pp + [np.minimum(rel, 3), n]))
names = ['const', 'cg own phase', 'cg other phase', 'cg own pre', 'cg other pre', 'p pos0', 'p pos1', 'p pos2', 'p pos3', 'c/o', 'size own'] + [f'd{d}' for d in (7, 11, 15, 19)]
for lab, A in (('LEAVE (my members)', LA), ('JOIN (ah members leaving ah)', JA)):
    A = np.concatenate(A)
    X = np.column_stack([np.ones(len(A)), A[:, 2:12]] + [(A[:, 0] == d) * 1.0 for d in (7, 11, 15, 19)])
    y = A[:, 1]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(lab, 'n', len(A), 'mean', y.mean().round(3))
    for nm, bb in zip(names, b): print(f'   {nm:16s} {bb:+.4f}')
