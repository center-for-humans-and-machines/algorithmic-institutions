"""v3: random local search on the surrogate mean over (T,A,B) for round 0 and (T,A,B) for rounds
1..S plus S, starting from v2's best (does a separate opening schedule help?)."""
import sys, numpy as np; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from screen import *
rng = np.random.default_rng(0)

def two(T0, A0, B0, T1, A1, B1, S):
    def f(c, v, p, t):
        if t > S: return zeros(c)
        T, A, B = (T0, A0, B0) if t == 0 else (T1, A1, B1)
        s = th.relu(T - c[..., -1])
        return th.where(s > 0, A + B * s, zeros(c))
    return f

def sc(x, seed=1, B=4000):
    r = ev(two(*x), B=B, seed=seed)
    return (r['mlp']['score'] + r['gru']['score']) / 2

x = np.array([14.3, 1.7, 1.56, 14.3, 1.7, 1.56, 8.])
step = np.array([1.5, 1.0, 0.3, 1.5, 1.0, 0.3, 2])
lo = np.array([8, 0, 0.3, 8, 0, 0.3, 2]); hi = np.array([20, 10, 4, 20, 10, 4, 16])
best = sc(x); print('start', round(best, 1))
for it in range(int(sys.argv[1])):
    y = np.clip(x + rng.normal(size=7) * step * (rng.random(7) < 0.4), lo, hi); y[6] = round(y[6])
    s = sc(y)
    if s > best + 0.5:
        s2 = (s + sc(y, seed=99)) / 2; b2 = (best + sc(x, seed=99)) / 2
        if s2 > b2:
            x, best = y, s; print(it, 'accept', np.round(x, 2), round(s, 1), round(s2, 1), flush=True)
print('final', np.round(x, 2), best)
