"""v1 games: at each decision, the share of my members who leave and of the other group's members who join me,
against the block's mean member payoff gap (mine - theirs) and my group's mean contribution and punishment."""
import numpy as np, pandas as pd
z = np.load('/tmp/v1_compact.npz'); c, p, m, v = [z[k].astype(np.float32) for k in ['c', 'p', 'mine', 'v']]
rows = []
for r in [3, 7, 11, 15, 19]:
    b = slice(r - 3, r + 1)
    mb, cb_, pb_, vb = m[:, :, b], c[:, :, b], p[:, :, b], v[:, :, b]
    def grp(mask):
        n = mask.sum(1)  # [G, 4]
        sc = (cb_ * mask).sum(1)
        share = 1.6 * (cb_ * mask).sum(1) / np.maximum((mask * vb).sum(1), 1)
        pay = (((20 - cb_ - pb_) + share[:, None, :]) * mask * vb).sum((1, 2)) / np.maximum((mask * vb).sum((1, 2)), 1)
        return n.mean(-1), sc.sum(-1) / np.maximum(n.sum(-1), 1), (pb_ * mask).sum((1, 2)) / np.maximum(n.sum(-1), 1), pay
    n0, c0, p0, pay0 = grp(mb); n1, c1, p1, pay1 = grp(1 - mb)
    leave = ((m[:, :, r] == 1) & (m[:, :, r + 1] == 0)).sum(1); join = ((m[:, :, r] == 0) & (m[:, :, r + 1] == 1)).sum(1)
    rows.append(pd.DataFrame(dict(r=r, n0=m[:, :, r].sum(1), leave=leave, join=join, c0=c0, c1=c1, p0=p0, p1=p1, gap=pay0 - pay1)))
X = pd.concat(rows); X['n1'] = 8 - X.n0
X = X[(X.n0 > 0) & (X.n1 > 0)]
X['lr'] = X.leave / X.n0; X['jr'] = X['join'] / X.n1
print('by payoff gap (my members minus theirs, per member-round in the block)')
print(X.groupby(pd.cut(X.gap, [-50, -6, -3, -1, 1, 3, 6, 50]), observed=True)[['lr', 'jr', 'c0', 'p0', 'p1']].mean().round(3).assign(n=X.groupby(pd.cut(X.gap, [-50, -6, -3, -1, 1, 3, 6, 50]), observed=True).size()))
print('by my mean contribution, rows; my punishment per member-round == 0 only')
Z = X[X.p0 == 0]
print(Z.groupby(pd.cut(Z.c0, [-1, 4, 7, 10, 13, 21]), observed=True)[['lr', 'jr', 'c1', 'gap']].mean().round(3))
print('by my size at the decision')
print(X.groupby('n0')[['lr', 'jr']].mean().round(3))
import numpy.linalg as la
A = np.c_[np.ones(len(X)), X.gap, X.c0, X.c1, X.p0, X.p1, X.n0]
for y in ['lr', 'jr']:
    b = la.lstsq(A, X[y].values, rcond=None)[0]
    print(y, dict(zip(['1', 'gap', 'c0', 'c1', 'p0', 'p1', 'n0'], b.round(4))))
