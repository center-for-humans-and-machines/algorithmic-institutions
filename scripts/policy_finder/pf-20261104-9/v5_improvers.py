"""Pooled sweeps: for laggards (c < 0.9 x others' mean) in within-phase rounds (1,2,5,6,9,10), does a
punishment now buy as much for members who already improved since the last round as for those who did
not? Also split by group size. OLS of c(t+3)-c(t) (still in group) and of staying after the next
decision on p_t (linear, points) with c, o, c(t-1), p(t-1), round dummies."""
import numpy as np
vers = ['pf-20261104-9', 'pf-20261104-9-v2', 'pf-20261104-9-v3', 'pf-20261104-9-v4']
rows = []
for v in vers:
    z = np.load(f'/tmp/pf/{v}.npz'); C = z['C'].astype(np.float32); V = z['V']; P = z['P'].astype(np.float32); G = z['G']
    m = (G == 0) & (V == 1)
    k = m.sum(2, keepdims=True)
    o = ((C * m).sum(2, keepdims=True) - C * m) / np.maximum(k - m, 1)
    for t in [1, 2, 5, 6, 9, 10]:
        sel = m[..., t] & m[..., t - 1] & (k[..., 0, t] >= 2)[..., None] & (C[..., t] < 0.9 * o[..., t])
        dnext = t + 4 - t % 4
        rows.append(np.column_stack([np.full(sel.sum(), t), C[..., t][sel], o[..., t][sel], P[..., t][sel], C[..., t - 1][sel],
                                     P[..., t - 1][sel], C[..., t + 3][sel], ((G[..., t + 3] == 0) & (V[..., t + 3] == 1))[sel],
                                     (G[..., dnext] == 0)[sel], np.broadcast_to(k[..., 0, t][..., None], sel.shape)[sel]]))
R = np.concatenate(rows); t, c, o, p, c1, p1, c3, g3, stay, n = R.T
print('n', len(R))
def fit(s, lab):
    X = np.column_stack([np.ones(s.sum()), c[s], o[s], c1[s], p1[s], p[s]] + [(t[s] == tt) * 1.0 for tt in (2, 5, 6, 9, 10)])
    s2 = s & (g3 == 1)
    X2 = np.column_stack([np.ones(s2.sum()), c[s2], o[s2], c1[s2], p1[s2], p[s2]] + [(t[s2] == tt) * 1.0 for tt in (2, 5, 6, 9, 10)])
    b, *_ = np.linalg.lstsq(X2, c3[s2] - c[s2], rcond=None)
    b2, *_ = np.linalg.lstsq(X, stay[s], rcond=None)
    print(f'{lab:40s} n{s.sum()//1000:5d}k  lift/pt {b[5]:+.4f}  stay/pt {b2[5]:+.4f}  mean p {p[s].mean():.1f}  share punished {(p[s]>0).mean():.2f}')
d = c - c1
for lab, s in [('all', np.ones_like(c, bool)), ('punished last round, improved >=3', (p1 > 0) & (d >= 3)),
               ('punished last round, improved 1-2', (p1 > 0) & (d >= 1) & (d < 3)),
               ('punished last round, not improved', (p1 > 0) & (d <= 0)),
               ('unpunished last round, improved >=3', (p1 == 0) & (d >= 3)),
               ('unpunished last round, not improved', (p1 == 0) & (d <= 0))]:
    fit(s, lab)
for nn in (2, 3, 4, 5, 6):
    fit(n == nn, f'group size {nn}')
