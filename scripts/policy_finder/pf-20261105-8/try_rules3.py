"""Credit for improvement: shortfall reduced by credit * rise since last round."""
import sys, numpy as np
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-8')
from surrogate2 import simulate
def rule(thr, slope, stop, credit):
    def f(C, P, t):
        c = C[:, :, -1]
        rise = np.maximum(c - C[:, :, -2], 0) if C.shape[2] > 1 else 0
        gap = np.maximum(thr - c - credit * rise, 0)
        return slope * gap * (t < stop)
    return f
for kw in [dict(thr=16, slope=1, stop=8, credit=0), dict(thr=16, slope=1, stop=8, credit=1),
           dict(thr=16, slope=1, stop=12, credit=1), dict(thr=16, slope=1.5, stop=10, credit=1),
           dict(thr=16, slope=1, stop=16, credit=2), dict(thr=18, slope=1.5, stop=12, credit=1)]:
    s, cm, pm = simulate(rule(**kw), E=600)
    print(kw, round(s, 1), np.round(cm[::4], 1), round(pm, 2))
