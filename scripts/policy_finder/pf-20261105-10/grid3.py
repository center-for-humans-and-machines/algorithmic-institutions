"""Tapering gap rule with a cap on the surrogate."""
import sys, numpy as np, itertools
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-10')
from simsur import simulate
def rule(k, target, stop, cap):
    def f(c, p, t):
        w = max(0.0, 1 - t / stop)
        return w * np.minimum(cap, k * np.maximum(0, target - c[..., -1]))
    return f
for k, target, stop, cap in [(2,15,10,30),(3,15,10,30),(4,15,10,30),(2,18,10,30),(2,20,10,30),(3,18,8,30),
                             (2,15,6,30),(2,15,14,30),(2,15,20,30),(3,15,10,10),(3,15,10,6),(4,18,10,8),(10,18,10,6),(10,20,8,5)]:
    s, se, cm, pm = simulate(rule(k, target, stop, cap), G=1500)
    print((k, target, stop, cap), round(s, 1), round(se, 1), cm[[0, 4, 8, 12, 16, 23]].round(1), round(pm.sum(), 1), flush=True)
