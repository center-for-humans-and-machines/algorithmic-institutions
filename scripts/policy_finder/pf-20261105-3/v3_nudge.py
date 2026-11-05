"""v3: high contributors (c 12-19) in rounds 3-14 of v2's sweep: change k rounds later by exact c and own p.
Also the others' mean contribution change (does a nudge on one player move the group?)."""
import sys; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-3')
import numpy as np
from v3_load import load
a = load(); c, p, v = a['c'], a['p'], a['v']
for k in [1, 4]:
    print(f'=== k={k}')
    for cc in range(12, 20):
        out = []
        for pl, ph in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 30)]:
            d = []; n = 0
            for t in range(3, 15):
                m = (c[..., t] == cc) & (p[..., t] >= pl) & (p[..., t] <= ph) & v[..., t] & v[..., t + k]
                d.append(c[..., t + k][m] - cc); n += m.sum()
            d = np.concatenate(d)
            out.append(f'p{pl}-{ph}: {d.mean():+5.2f} ({n:6d})' if n > 100 else f'p{pl}-{ph}:   nan        ')
        print(f'c={cc}', ' | '.join(out))
