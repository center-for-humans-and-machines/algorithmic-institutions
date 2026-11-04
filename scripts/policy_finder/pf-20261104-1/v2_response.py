"""v2: on v1's sweep (our group = 0): contribution response to punishment by
contribution / shortfall bins, and leaving/joining at switch decisions."""
import numpy as np, pandas as pd
a = np.load('/tmp/v2_arrays.npz')
C, P, G, V = [a[k].astype(np.float32) for k in 'CPGV']
prm = a['prm']
Pn, E, N, R = C.shape
# mates mean per round in own group
ours = (G == 0)
def mates_mean(mask):
    n = mask.sum(2, keepdims=True); s = (C * mask).sum(2, keepdims=True)
    return (s - C * mask) / np.maximum(n - mask, 1), n
mm0, n0 = mates_mean(ours * V)
rows = []
for r in range(R - 1):
    if r % 4 == 3: continue
    m = ours[..., r] & ours[..., r + 1] & (V[..., r] > 0) & (V[..., r + 1] > 0)
    rows.append(pd.DataFrame({'r': r, 'c': C[..., r][m], 'cn': C[..., r + 1][m], 'p': P[..., r][m],
                              'mm': mm0[..., r][m], 'pt': np.broadcast_to(np.arange(Pn)[:, None, None], m.shape)[m]}))
d = pd.concat(rows); d['dc'] = d.cn - d.c; d['gap'] = d.mm - d.c
d['cb'] = pd.cut(d.c, [-1, 4, 9, 14, 20]); d['pb'] = pd.cut(d.p, [-1, 0, 2, 5, 10, 30])
print('dc by contribution bin x punishment bin (mean, n/1000)')
print(d.groupby(['cb', 'pb'], observed=True).dc.agg(['mean', 'size']).assign(size=lambda x: (x['size'] / 1000).round(0)).unstack().round(2).to_string())
d['gb'] = pd.cut(d.gap, [-21, -3, 0, 3, 6, 10, 21])
print('dc by gap bin x punishment bin')
print(d.groupby(['gb', 'pb'], observed=True).dc.mean().unstack().round(2).to_string())
import numpy.linalg as la
X = np.c_[np.ones(len(d)), d.c, d.mm, d.p, np.minimum(d.p, 5)]
b = la.lstsq(X, d.cn, rcond=None)[0]; print('cn ~ 1 c mm p min(p,5):', b.round(3))
# leaving at decisions
rows = []
for r in [3, 7, 11, 15, 19]:
    blk = slice(r - 3, r + 1)
    stay = ours[..., blk].all(-1)
    m = stay & (V[..., r] > 0)
    left = (G[..., r + 1] != 0)
    rows.append(pd.DataFrame({'r': r, 'psum': P[..., blk].sum(-1)[m], 'plast': P[..., r][m], 'c': C[..., blk].mean(-1)[m],
                              'mm': mm0[..., blk].mean(-1)[m], 'n': np.broadcast_to(n0[..., r], m.shape)[m], 'left': left[m]}))
L = pd.concat(rows)
L['pb'] = pd.cut(L.psum, [-1, 0, 4, 10, 20, 40, 120])
print('P(leave) by block punishment sum'); print(L.groupby('pb', observed=True).left.agg(['mean', 'size']).round(3).to_string())
L['plb'] = pd.cut(L.plast, [-1, 0, 2, 5, 10, 30])
print('P(leave) by last-round punishment, within psum 4-20'); x = L[(L.psum > 4) & (L.psum <= 20)]
print(x.groupby('plb', observed=True).left.agg(['mean', 'size']).round(3).to_string())
L['cb'] = pd.cut(L.c, [-1, 4, 9, 14, 20])
print('P(leave) by own c bin x pb'); print(L.groupby(['cb', 'pb'], observed=True).left.mean().unstack().round(3).to_string())
