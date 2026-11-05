"""Try rule shapes on the nonparametric surrogate."""
import sys, numpy as np
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-8')
from surrogate2 import simulate
def rule(thr, slope, stop, rel=0.0):
    def f(C, P, t):
        c = C[:, :, -1]
        oth = (c.sum(1, keepdims=True) - c) / 3
        std = (1 - rel) * thr + rel * np.minimum(oth, thr)
        return slope * np.maximum(std - c, 0) * (t < stop)
    return f
s, cm, pm = simulate(lambda C, P, t: np.zeros(C.shape[:2]))
print('zero', round(s, 1), np.round(cm[::4], 1), round(pm, 2))
for thr in [8, 12, 16]:
    for slope in [0.25, 0.5, 1.0, 2.0]:
        for stop in [8, 16, 24]:
            s, cm, pm = simulate(rule(thr, slope, stop))
            print(thr, slope, stop, round(s, 1), np.round(cm[::4], 1), round(pm, 2))
