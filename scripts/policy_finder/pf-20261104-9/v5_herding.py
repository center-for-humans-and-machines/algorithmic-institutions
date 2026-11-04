"""Pooled sweeps: herding. For UNPUNISHED members of my group in rounds 0-10 (not pre-decision), the
change c(t+3)-c(t) (still in group) and staying after the next decision against the punishment the
OTHER members got this round: total points (bins), and the number of others punished, with c, o,
group size and round dummies as controls; split by c / o class."""
import numpy as np
vers = ['pf-20261104-9', 'pf-20261104-9-v2', 'pf-20261104-9-v3', 'pf-20261104-9-v4']
rows = []
for v in vers:
    z = np.load(f'/tmp/pf/{v}.npz'); C = z['C'].astype(np.float32); V = z['V']; P = z['P'].astype(np.float32); G = z['G']
    m = (G == 0) & (V == 1)
    k = m.sum(2, keepdims=True)
    o = ((C * m).sum(2, keepdims=True) - C * m) / np.maximum(k - m, 1)
    Pm = P * m
    po = Pm.sum(2, keepdims=True) - Pm
    npo = (Pm > 0).sum(2, keepdims=True) - (Pm > 0)
    for t in [0, 1, 2, 4, 5, 6, 8, 9, 10]:
        sel = m[..., t] & (k[..., 0, t] >= 2)[..., None] & (P[..., t] == 0)
        dnext = t + 4 - t % 4
        B = lambda a: np.broadcast_to(a, sel.shape)[sel]
        rows.append(np.column_stack([np.full(sel.sum(), t), C[..., t][sel], o[..., t][sel], po[..., t][sel], npo[..., t][sel],
                                     C[..., t + 3][sel], ((G[..., t + 3] == 0) & (V[..., t + 3] == 1))[sel],
                                     (G[..., dnext] == 0)[sel], B(k[..., 0, t][..., None])]))
R = np.concatenate(rows); t, c, o, po, npo, c3, g3, stay, n = R.T
np.save('/tmp/pf/herd_rows.npy', R)
r = c / np.maximum(o, 0.5)
bins = [(1, 11), (11, 31), (31, 61), (61, 241)]
for lo, hi in [(0, 0.8), (0.8, 1.1), (1.1, 9)]:
    s = (r >= lo) & (r < hi)
    def X_(s):
        return np.column_stack([np.ones(s.sum()), c[s], o[s], n[s]] + [(t[s] == tt) * 1.0 for tt in (1, 2, 4, 5, 6, 8, 9, 10)]
                               + [((po[s] >= a) & (po[s] < b)) * 1.0 for a, b in bins] + [po[s] / (n[s] - 1)])
    s2 = s & (g3 == 1)
    b, *_ = np.linalg.lstsq(X_(s2)[:, :-1], c3[s2] - c[s2], rcond=None)
    b2, *_ = np.linalg.lstsq(X_(s)[:, :-1], stay[s], rcond=None)
    print(f'c/o {lo}-{hi} n{s.sum()//1000}k  others-points bins ' + ' '.join(f'{a}-{bb-1}: {x:+.2f}/{y:+.3f}' for (a, bb), x, y in zip(bins, b[-4:], b2[-4:])))
    bl, *_ = np.linalg.lstsq(np.column_stack([X_(s2)[:, :12], po[s2]]), c3[s2] - c[s2], rcond=None)
    print(f'      linear per point others got: lift {bl[-1]:+.4f}')
