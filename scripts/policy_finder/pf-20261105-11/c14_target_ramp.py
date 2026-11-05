"""v3: does a target that rises over the punishment window beat v2's fixed target?
T_t = min(20, T0 + r * t) for t <= S; punishment A + B * (T_t - c) below it."""
import sys, itertools, numpy as np; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from screen import *


def ramp(T0, r, A, B, S):
    def f(c, v, p, t):
        if t > S: return zeros(c)
        T = min(20.0, T0 + r * t)
        s = th.relu(T - c[..., -1])
        return th.where(s > 0, A + B * s, zeros(c))
    return f


show('v2 best', ev(ramp(14.3, 0, 1.7, 1.56, 8)))
res = []
for T0, r, S, B in itertools.product([11, 13, 14.3], [0, 0.5, 1.0], [6, 8, 10], [1.2, 1.56, 2.2]):
    x = ev(ramp(T0, r, 1.7, B, S), B=3000)
    m = (x['mlp']['score'] + x['gru']['score']) / 2
    res.append((m, x['mlp']['score'], x['gru']['score'], T0, r, S, B))
res.sort(reverse=True)
for row in res[:15]:
    print(np.round(row, 2))
print('best with r=0:', [np.round(r_, 2) for r_ in res if r_[4] == 0][:3])
