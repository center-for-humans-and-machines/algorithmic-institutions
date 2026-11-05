"""Grid of rule shapes on the surrogate."""
import sys, time, numpy as np
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-10')
from simsur import simulate
def gap(k, target, stop):
    return lambda c, p, t: (t < stop) * k * np.maximum(0, target - c[..., -1])
def flat(amount, target, stop):
    return lambda c, p, t: (t < stop) * amount * (c[..., -1] < target)
t0 = time.time()
res = []
for stop in [8, 16, 23]:
    for target in [10, 15, 20]:
        for k in [0.25, 0.5, 1.0]:
            s, se, cm, pm = simulate(gap(k, target, stop), G=1000)
            res.append(('gap', k, target, stop, s)); print('gap', k, target, stop, round(s,1), round(se,1), round(cm[-1],1), round(pm.sum(),1), flush=True)
        for a in [1, 2, 4]:
            s, se, cm, pm = simulate(flat(a, target, stop), G=1000)
            print('flat', a, target, stop, round(s,1), round(se,1), round(cm[-1],1), round(pm.sum(),1), flush=True)
print(time.time()-t0)
