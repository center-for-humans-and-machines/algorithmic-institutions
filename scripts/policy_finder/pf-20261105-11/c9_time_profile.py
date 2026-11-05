"""v3: which rounds should be punished how hard? Coordinate search over a per-round multiplier
k_t (rounds 0..12) of v2's schedule, on the mean of the two surrogates."""
import sys, numpy as np; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from screen import *

def prof(k, T=14.3, A=1.7, B=1.56):
    def f(c, v, p, t):
        if t >= len(k) or k[t] == 0:
            return zeros(c)
        s = th.relu(T - c[..., -1])
        return th.where(s > 0, (A + B * s) * k[t], zeros(c))
    return f

def score(k, seed=1):
    r = ev(prof(k), B=4000, seed=seed)
    return (r['mlp']['score'] + r['gru']['score']) / 2

k = [1.0] * 9 + [0.0] * 4
best = score(k); print('start', best)
for sweep in range(2):
    for t in range(13):
        cands = [0, 0.5, 1.0, 1.5, 2.5] if t > 0 else [0.5, 1, 1.5, 2.5, 4]
        res = []
        for x in cands:
            kk = list(k); kk[t] = x
            res.append(score(kk))
        j = int(np.argmax(res))
        if res[j] > best + 0.8:
            k[t] = cands[j]; best = res[j]
        print(sweep, t, 'cands', np.round(res, 1), '-> k', k, round(best, 1), flush=True)
