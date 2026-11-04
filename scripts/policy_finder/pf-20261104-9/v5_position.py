"""Pooled sweeps: laggards (c < 0.9 x others' mean), rounds 0-10: lift per point (c in round 13, i.e.
long-run, for members in my group then) and staying after the next decision, by phase position
(0,1,2) and dose bin. OLS with c, o, p(t-1), group size, round dummies."""
import numpy as np
vers = ['pf-20261104-9', 'pf-20261104-9-v2', 'pf-20261104-9-v3', 'pf-20261104-9-v4']
rows = []
for v in vers:
    z = np.load(f'/tmp/pf/{v}.npz'); C = z['C'][::2].astype(np.float32); V = z['V'][::2]; P = z['P'][::2].astype(np.float32); G = z['G'][::2]
    m = (G == 0) & (V == 1)
    k = m.sum(2, keepdims=True)
    o = ((C * m).sum(2, keepdims=True) - C * m) / np.maximum(k - m, 1)
    for t in [0, 1, 2, 4, 5, 6, 8, 9, 10]:
        sel = m[..., t] & (k[..., 0, t] >= 2)[..., None] & (C[..., t] < 0.9 * o[..., t])
        dnext = t + 4 - t % 4
        B = lambda a: np.broadcast_to(a, sel.shape)[sel]
        pprev = P[..., t - 1][sel] if t > 0 else np.zeros(sel.sum())
        rows.append(np.column_stack([np.full(sel.sum(), t), C[..., t][sel], o[..., t][sel], P[..., t][sel], pprev,
                                     C[..., 13][sel], ((G[..., 13] == 0) & (V[..., 13] == 1))[sel],
                                     (G[..., dnext] == 0)[sel], B(k[..., 0, t][..., None])]))
R = np.concatenate(rows); t, c, o, p, p1, c13, g13, stay, n = R.T
pos = t % 4
for ps in (0, 1, 2):
    s = pos == ps
    def X_(s, lin):
        base = [np.ones(s.sum()), c[s], o[s], p1[s], n[s]] + [(t[s] == tt) * 1.0 for tt in (4, 8, 5, 9, 6, 10) if tt % 4 == ps]
        if lin: return np.column_stack(base + [p[s]])
        return np.column_stack(base + [((p[s] >= a) & (p[s] < b)) * 1.0 for a, b in ((1, 11), (11, 21), (21, 31))])
    s2 = s & (g13 == 1)
    bl, *_ = np.linalg.lstsq(X_(s2, 1), c13[s2] - c[s2], rcond=None)
    bs, *_ = np.linalg.lstsq(X_(s, 1), stay[s], rcond=None)
    bb, *_ = np.linalg.lstsq(X_(s2, 0), c13[s2] - c[s2], rcond=None)
    bbs, *_ = np.linalg.lstsq(X_(s, 0), stay[s], rcond=None)
    print(f'pos {ps} n{s.sum()//1000}k mean p {p[s].mean():.1f}: per point lift(c13) {bl[-1]:+.4f} stay {bs[-1]:+.4f} | bins 1-10/11-20/21-30 lift ' +
          ' '.join(f'{x:+.2f}' for x in bb[-3:]) + ' stay ' + ' '.join(f'{x:+.3f}' for x in bbs[-3:]))
