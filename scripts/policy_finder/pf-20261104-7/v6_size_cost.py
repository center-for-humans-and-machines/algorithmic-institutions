"""v6: member cost per punishment point by group size at decisions 7, 11, 15. OLS within each size
of the size change on my mean contribution and mean punishment per member (rounds d-3..d) and the
other group's mean cg; report the punishment slope and the slope per point spent (/size). Also the
contribution gain of the group one window later per point spent, by size."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-7')
from v5_panel import load
r = load(sys.argv[1] if len(sys.argv) > 1 else 'pf-20261104-7-v5')
c, p, v, g = r['c'].astype(float), r['p'].astype(float), r['v'].astype(float), r['g'].astype(float)
def cg(mask):
    n = (mask * v).sum(1)
    return np.where(n > 0, (1.6 * (c * mask * v).sum(1) - (p * mask * v).sum(1)) / np.maximum(n, 1), 0)
co = cg(1 - g)
size = g.sum(1)
cm = (c * g).sum(1) / np.maximum(size, 1)
pmm = (p * g).sum(1) / np.maximum(size, 1)
for d in [7, 11, 15]:
    w = slice(d - 3, d + 1)
    print(f'=== decision {d}')
    for n in range(1, 9):
        m = (size[:, w] == n).all(1)
        if m.sum() < 500: continue
        dn = size[m, d + 1] - n
        X = np.column_stack([np.ones(m.sum()), cm[m, w].mean(1), pmm[m, w].mean(1), co[m, w].mean(1)])
        b = np.linalg.lstsq(X, dn, rcond=None)[0]
        print(f'size {n}: N {m.sum():6d}  mean p/member {pmm[m, w].mean():5.2f}  dn/c {b[1]:+.3f}  dn/p {b[2]:+.3f}  per point spent {b[2]/n:+.4f}')
