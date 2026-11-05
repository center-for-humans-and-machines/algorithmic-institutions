"""v3: does a repeated nudge wear off? c 13-17 in rounds 6-18, own p in {0, 1, 2, 3-4}, split by the
number of earlier rounds in which the player was punished at 1-4 points (small doses only); change 3 rounds later."""
import sys; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-3')
import numpy as np
from v3_load import load
a = load(); c, p, v = a['c'], a['p'], a['v']
small = ((p >= 1) & (p <= 4)).astype(np.int16)
ns = np.cumsum(small, -1)
big = np.cumsum((p >= 5).astype(np.int16), -1)
k = 3
for nl, nh in [(0, 0), (1, 2), (3, 5), (6, 30)]:
    out = []
    for pl, ph in [(0, 0), (1, 1), (2, 2), (3, 4)]:
        d = []
        for t in range(6, 19):
            m = (c[..., t] >= 13) & (c[..., t] <= 17) & (ns[..., t - 1] >= nl) & (ns[..., t - 1] <= nh) \
                & (p[..., t] >= pl) & (p[..., t] <= ph) & v[..., t] & v[..., t + k]
            d.append(c[..., t + k][m] - c[..., t][m])
        d = np.concatenate(d)
        out.append(f'{d.mean():+5.2f} ({len(d):6d})' if len(d) > 200 else '  nan         ')
    print(f'earlier small doses {nl}-{nh}:', ' | '.join(out))
