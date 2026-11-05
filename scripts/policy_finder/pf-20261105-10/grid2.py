"""Stop round and tapering of the gap rule on the surrogate; relative targets."""
import sys, numpy as np
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-10')
from simsur import simulate
def gap(k, target, stop, taper=False, rel=False):
    def f(c, p, t):
        x = c[..., -1]
        tg = target
        if rel:  # target: others' mean minus `target`
            tg = (x.sum(1, keepdims=True) - x) / 3 - target
        w = max(0.0, 1 - t / stop) if taper else float(t < stop)
        return w * k * np.maximum(0, tg - x)
    return f
for args in [(1, 15, 8), (1, 15, 4), (1, 15, 6), (1, 15, 11), (2, 15, 6), (1.5, 15, 8), (1, 17, 6),
             (1, 15, 12, True), (1, 15, 16, True), (2, 15, 10, True), (1, 18, 12, True), (2, 12, 10, True)]:
    s, se, cm, pm = simulate(gap(*args), G=1500)
    print(args, round(s, 1), round(se, 1), cm[[0, 4, 8, 12, 16, 23]].round(1), round(pm.sum(), 1), flush=True)
for args in [(1, 0, 8), (2, 0, 8), (1, 2, 8), (1, 0, 16, True), (2, 0, 12, True)]:
    s, se, cm, pm = simulate(gap(args[0], args[1], args[2], len(args) > 3, rel=True), G=1500)
    print('rel', args, round(s, 1), round(se, 1), cm[[0, 4, 8, 12, 16, 23]].round(1), round(pm.sum(), 1), flush=True)
