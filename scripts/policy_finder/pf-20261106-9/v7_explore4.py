# v7: per opponent, by quintile of the opening punishment (our rounds 0-1 punishment per original
# member, per game), our members in rounds 4-7 / 8-23, our contribution in 8-23, pool 0-23, and
# the pool split into rounds 0-1 / 2-23. Over v6's sweep.
import numpy as np
d = np.load('/tmp/pf-20261106-9-v6_sweep.npz')
c = d['c'].astype(float); p = d['p'].astype(float); g = d['g']; ours = d['ours']; opp = d['opp']
mine = (g == ours[:, None, None])
pool = 1.6 * (c * mine).sum(1) - (p * mine).sum(1)
nm = mine.sum(1)
op = (p[:, :, :2] * mine[:, :, :2]).sum((1, 2)) / 4
cl = (c[:, :, 8:] * mine[:, :, 8:]).sum((1, 2)) / np.maximum(nm[:, 8:].sum(1), 1)
for o in range(1, 5):
    m = opp == o
    q = np.quantile(op[m], [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(op[m], q)
    print('opp', o)
    for k in range(5):
        s = b == k
        print(f'  open {op[m][s].mean():5.1f}  n4-7 {nm[m][s][:, 4:8].mean():.2f} n8-23 {nm[m][s][:, 8:].mean():.2f} c8-23 {cl[m][s].mean():5.2f}'
              f'  pool {pool[m][s].mean():5.1f} (0-1 {pool[m][s][:, :2].mean():5.1f}, 2-23 {pool[m][s][:, 2:].mean():5.1f})')
