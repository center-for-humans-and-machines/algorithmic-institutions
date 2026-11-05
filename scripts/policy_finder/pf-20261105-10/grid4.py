"""Offset (base punishment for any shortfall), taper shape and forgiveness of risers, on the surrogate."""
import sys, numpy as np
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-10')
from simsur import simulate
def rule(k, target, stop, base=0.0, shape=1.0, forgive=0.0):
    def f(c, p, t):
        x = c[..., -1]
        w = max(0.0, 1 - t / stop) ** shape
        gap = np.maximum(0, target - x)
        pun = (gap > 0) * (base + k * gap)
        if forgive and t > 0:
            rise = np.maximum(0, x - c[..., -2])
            pun = np.maximum(0, pun - forgive * rise)
        return w * pun
    return f
cases = [dict(k=3, target=15, stop=10), dict(k=2, target=15, stop=10, base=2), dict(k=1, target=15, stop=10, base=5),
         dict(k=2, target=15, stop=14, shape=2), dict(k=3, target=15, stop=16, shape=2), dict(k=3, target=15, stop=10, shape=0.5),
         dict(k=3, target=15, stop=10, forgive=1), dict(k=3, target=15, stop=14, forgive=2), dict(k=3, target=13, stop=10),
         dict(k=3, target=16, stop=8)]
for a in cases:
    s, se, cm, pm = simulate(rule(**a), G=2000)
    print(a, round(s, 1), round(se, 1), cm[[0, 4, 8, 12, 16, 23]].round(1), round(pm.sum(), 1), flush=True)
