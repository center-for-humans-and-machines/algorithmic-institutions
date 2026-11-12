"""v4: how many members, and how many points, the v4 rule would hit at round 4 (state from v3's sweep, which is
param-independent up to round 4), for a few candidate settings; with the per-point estimates of explorations 3/5
this gives a rough expected gain."""
import numpy as np
D = np.load('/tmp/v3sweep.npz')
c, g, v, mg = D['c'], D['g'], D['v'], D['mine_g']
mine = (g == mg[:, None, None])[:, :, 4] & (v[:, :, 4] > 0)
c4 = c[:, :, 4]
print('own size r4 mean', mine.sum(1).mean())
for lo, thr in [(2, 8), (2, 12), (2, 16), (4, 10), (0, 12)]:
    nf = ((c4 <= lo) & mine).sum(1).mean(); nm = ((c4 > lo) & (c4 <= thr) & mine).sum(1).mean()
    print(f'lo {lo} thr {thr}: free riders {nf:.2f}, middling {nm:.2f} per game')
