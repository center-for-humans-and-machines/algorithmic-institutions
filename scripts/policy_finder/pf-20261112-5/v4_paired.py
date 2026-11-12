"""Final runs of a version: per-episode lead per point, paired differences against the peak (same seeds)."""
import numpy as np, sys
z = np.load(sys.argv[1]); c, p, m, v = [z[k].astype(np.float32) for k in ['c', 'p', 'mine', 'v']]
pt, seat = z['point'].astype(int), z['seat'].astype(int)
pool = 1.6 * c * v - p
lead = ((pool * m).sum(1) - (pool * (1 - m)).sum(1)).mean(1)
own = (pool * m).sum(1).mean(1)
mem = m.sum(1).mean(1)
print('round0 c identical across points?', [np.abs(c[(pt == a) & (seat == 0), :, 0] - c[(pt == pt.max()) & (seat == 0), :, 0]).mean() for a in np.unique(pt)])
ref = -1
for a in np.unique(pt):
    s = pt == a; r = pt == ref
    dd = lead[s] - lead[r]
    print(a, 'lead %.2f se %.2f | diff vs gp %.2f se %.2f | own %.1f mem %.2f' % (lead[s].mean(), lead[s].std() / np.sqrt(s.sum()), dd.mean(), dd.std() / np.sqrt(s.sum()), own[s].mean(), mem[s].mean()))
