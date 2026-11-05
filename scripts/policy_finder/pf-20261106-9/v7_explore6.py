# v7: what are low-shortfall games? By tercile of round-0 group mean m0 and of D (total shortfall
# below the mean): n games, opening punishment per member (rounds 0-1), contribution in rounds
# 8-23, pool. Over v6's sweep.
import numpy as np
d = np.load('/tmp/pf-20261106-9-v6_sweep.npz')
c = d['c'].astype(float); p = d['p'].astype(float); g = d['g']; ours = d['ours']; opp = d['opp']
mine = (g == ours[:, None, None])
pool = (1.6 * (c * mine).sum(1) - (p * mine).sum(1)).mean(1)
c0 = c[:, :, 0]; m0 = (c0 * mine[:, :, 0]).sum(1) / 4
D = (np.clip(m0[:, None] - c0, 0, None) * mine[:, :, 0]).sum(1)
op = (p[:, :, :2] * mine[:, :, :2]).sum((1, 2)) / 4
nm = mine.sum(1)
cl = (c[:, :, 8:] * mine[:, :, 8:]).sum((1, 2)) / np.maximum(nm[:, 8:].sum(1), 1)
qm = np.quantile(m0, [1 / 3, 2 / 3]); qd = np.quantile(D, [1 / 3, 2 / 3])
print('m0 bounds', qm, 'D bounds', qd, 'corr', np.corrcoef(m0, D)[0, 1])
for i in range(3):
    for j in range(3):
        s = (np.digitize(m0, qm) == i) & (np.digitize(D, qd) == j)
        print(f'm0 t{i} ({m0[s].mean():5.2f}) D t{j} ({D[s].mean():5.1f}): share {s.mean():.3f} open {op[s].mean():5.1f} c8-23 {cl[s].mean():5.2f} pool {pool[s].mean():5.1f}')
# within-group spread of c0: min and max
print('c0 min/max by m0 tercile', [(c0 * 1)[np.digitize(m0, qm) == i].min(1).mean() for i in range(3)])
