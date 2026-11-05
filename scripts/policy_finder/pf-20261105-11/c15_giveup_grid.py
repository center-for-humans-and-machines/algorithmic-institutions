"""v3: give-up (stop punishing a player after n punished rounds) x window S x target T."""
import sys, itertools, numpy as np; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from screen import *

def giveup(T, A, B, S, n):
    def f(c, v, p, t):
        if t > S: return zeros(c)
        s = th.relu(T - c[..., -1])
        x = th.where(s > 0, A + B * s, zeros(c))
        if t == 0: return x
        k = (p > 0).float().sum(-1)
        return th.where(k >= n, zeros(c), x)
    return f

res = []
for T, S, n, B in itertools.product([13, 14.3, 16], [8, 11, 14], [3, 4, 5, 6, 99], [1.56, 2.2]):
    x = ev(giveup(T, 1.7, B, S, n), B=3000)
    res.append(((x['mlp']['score'] + x['gru']['score']) / 2, x['mlp']['score'], x['gru']['score'], T, S, n, B))
res.sort(reverse=True)
for r in res[:15]: print(np.round(r, 1))
for n in [3, 4, 5, 6, 99]:
    print('best with n=', n, np.round(max(r for r in res if r[5] == n), 1))
