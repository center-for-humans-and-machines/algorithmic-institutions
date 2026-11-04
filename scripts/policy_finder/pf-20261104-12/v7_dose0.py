"""Dose-response at round 0 (our group = agent_group 0) pooled over the v3-v6 sweeps:
round-1 contribution and leaving at decision 3 by round-0 contribution bin and fine
punishment bin, with a regression controlling for mates' round-0 mean."""
import numpy as np, sys
rows = []
for v in ['v3','v4','v5','v6']:
    z = np.load(f'/tmp/pf-20261104-12-{v}.npz')
    c, p, g = z['c'].astype(np.float32), z['p'].astype(np.float32), z['g']
    our = g[..., 0] == 0
    c0 = c[..., 0]; p0 = p[..., 0]; c1 = c[..., 1]
    n0 = our.sum(-1, keepdims=True)
    mates = ((c0 * our).sum(-1, keepdims=True) - c0) / np.maximum(n0 - 1, 1)
    leave = (g[..., 4] != g[..., 3]).astype(np.float32)
    c13 = c[..., 1:4].mean(-1)
    cl = c[..., 20:24].mean(-1)
    m = our
    rows.append(np.stack([c0[m], p0[m], c1[m], mates[m], leave[m], c13[m], cl[m], p[..., 1][m]], 1))
X = np.concatenate(rows); print(X.shape)
c0, p0, c1, mates, leave, c13, cl, p1 = X.T
bins = [(0,0),(1,2),(3,5),(6,10),(11,15),(16,20),(21,25),(26,29),(30,30)]
for lo, hi in [(0,2),(3,5),(6,8)]:
    sel = (c0 >= lo) & (c0 <= hi)
    print(f'c0 {lo}-{hi}')
    for a, b in bins:
        s = sel & (p0 >= a) & (p0 <= b)
        if s.sum() < 200: continue
        print(f'  p {a:2d}-{b:2d} n={s.sum():7d} mates={mates[s].mean():5.2f} c1={c1[s].mean():5.2f} c1-3={c13[s].mean():5.2f} leave={leave[s].mean():.3f} p1={p1[s].mean():5.1f}')
# regression c1 on p0 dummies with mates, c0 controls among c0<=8
sel = c0 <= 8
D = [((p0 >= a) & (p0 <= b)).astype(float) for a, b in bins[1:]]
Z = np.column_stack([np.ones(sel.sum())] + [d[sel] for d in D] + [mates[sel], c0[sel]])
for y, nm in [(c1, 'c1'), (leave, 'leave'), (c13, 'c1-3')]:
    b = np.linalg.lstsq(Z, y[sel], rcond=None)[0]
    print(nm, ' '.join(f'{a}-{bb}:{x:+.3f}' for (a, bb), x in zip(bins[1:], b[1:len(bins)])))
