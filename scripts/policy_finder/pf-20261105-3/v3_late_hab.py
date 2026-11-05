"""v3: (a) nudges on c 12-18 by game phase (rounds 3-8, 9-14, 15-20): change 3 rounds later;
(b) habituation: low contributors (c<=4) rounds 2-10, response 2 rounds later to own p by past summed p."""
import sys; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-3')
import numpy as np
from v3_load import load
a = load(); c, p, v = a['c'], a['p'], a['v']
k = 3
print('(a) c 12-18, change 3 rounds later; columns p=0 | 1 | 2 | 3-5')
for lo, hi in [(3, 9), (9, 15), (15, 21)]:
    for cl, ch in [(12, 14), (15, 16), (17, 18)]:
        out = []
        for pl, ph in [(0, 0), (1, 1), (2, 2), (3, 5)]:
            d = []
            for t in range(lo, hi):
                m = (c[..., t] >= cl) & (c[..., t] <= ch) & (p[..., t] >= pl) & (p[..., t] <= ph) & v[..., t] & v[..., t + k]
                d.append(c[..., t + k][m] - c[..., t][m])
            d = np.concatenate(d)
            out.append(f'{d.mean():+5.2f} ({len(d):6d})' if len(d) > 200 else '  nan         ')
        print(f'rounds {lo}-{hi-1} c {cl}-{ch}:', ' | '.join(out))
cum = np.cumsum(p, -1)
print('(b) c<=4, rounds 2-10, change 2 rounds later; columns own p 0 | 1-5 | 6-12 | 13-20 | 21-30')
for pl, ph in [(0, 0), (1, 20), (21, 45), (46, 80), (81, 1000)]:
    out = []
    for ql, qh in [(0, 0), (1, 5), (6, 12), (13, 20), (21, 30)]:
        d = []
        for t in range(2, 11):
            m = (c[..., t] <= 4) & (cum[..., t - 1] >= pl) & (cum[..., t - 1] <= ph) & (p[..., t] >= ql) & (p[..., t] <= qh) & v[..., t] & v[..., t + 2]
            d.append(c[..., t + 2][m] - c[..., t][m])
        d = np.concatenate(d)
        out.append(f'{d.mean():+5.2f} ({len(d):5d})' if len(d) > 200 else '  nan        ')
    print(f'past {pl:3d}-{ph:4d}:', ' | '.join(out))
