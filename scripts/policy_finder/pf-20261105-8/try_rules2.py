"""Variants on the surrogate: relative standard, taper, flat amount."""
import sys, numpy as np
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-8')
from surrogate2 import simulate
def rule(thr, slope, stop, rel=0.0, taper=False, flat=0.0, cap=30):
    def f(C, P, t):
        c = C[:, :, -1]
        oth = (c.sum(1, keepdims=True) - c) / 3
        std = (1 - rel) * thr + rel * np.minimum(oth, thr)
        gap = np.maximum(std - c, 0)
        w = max(0.0, 1 - t / stop) if taper else float(t < stop)
        return np.minimum(slope * gap + flat * (gap > 0), cap) * w
    return f
cases = {
 'base 16/1/8': dict(thr=16, slope=1, stop=8),
 'base 16/1/12': dict(thr=16, slope=1, stop=12),
 'base 18/1/8': dict(thr=18, slope=1, stop=8),
 'base 16/0.75/10': dict(thr=16, slope=0.75, stop=10),
 'taper 16/1/16': dict(thr=16, slope=1, stop=16, taper=True),
 'taper 16/1.5/14': dict(thr=16, slope=1.5, stop=14, taper=True),
 'rel 16/1/8': dict(thr=16, slope=1, stop=8, rel=1),
 'rel 16/1/16': dict(thr=16, slope=1, stop=16, rel=1),
 'cap8 16/1.5/10': dict(thr=16, slope=1.5, stop=10, cap=8),
 'flat3 16/0.5/10': dict(thr=16, slope=0.5, stop=10, flat=3),
}
for k, kw in cases.items():
    s, cm, pm = simulate(rule(**kw), E=600)
    print(k, round(s, 1), np.round(cm[::4], 1), round(pm, 2))
