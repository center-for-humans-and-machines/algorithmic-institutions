"""v7: per-value dose curves at round 4 (pooled v3-v6) and how many members per game sit at each value."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import v7_r4_fits as F
c4, here = F.c4, F.here
for a in range(0, 8):
    print('c', a, 'members per game at round 4', ((c4 == a) & here).sum(1).mean().round(3))
F.fit([(f'c{a}', (c4 == a)) for a in range(1, 6)] + [('c0', c4 == 0), ('c6-7', (c4 >= 6) & (c4 <= 7))],
      title='by value', ys=('lead', 'ownc'))
F.fit([('c1-2', (c4 >= 1) & (c4 <= 2)), ('c3', c4 == 3), ('c4-5', (c4 >= 4) & (c4 <= 5)), ('c0', c4 == 0)],
      title='1-2 / 3 / 4-5', ys=('lead',))
