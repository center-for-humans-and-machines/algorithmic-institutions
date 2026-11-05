"""v3: variants of give-up: total count (n), consecutive count (reset when a player is not
punished), and a smooth decay d^k of the fine with the count of earlier punished rounds."""
import sys, numpy as np; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from screen import *


def var(T, A, B, S, mode, n):
    def f(c, v, p, t):
        if t > S: return zeros(c)
        s = th.relu(T - c[..., -1])
        x = th.where(s > 0, A + B * s, zeros(c))
        if t == 0: return x
        pun = (p > 0).float()
        if mode == 'total':
            return th.where(pun.sum(-1) >= n, zeros(c), x)
        if mode == 'decay':
            return x * n ** pun.sum(-1)
        if mode == 'consec':
            # length of the current run of punished rounds, ending at t-1
            run = th.zeros_like(x)
            alive = th.ones_like(x)
            for j in range(pun.shape[-1] - 1, -1, -1):
                alive = alive * pun[..., j]
                run = run + alive
            return th.where(run >= n, zeros(c), x)
    return f


def both(rule, seeds=(21, 22)):
    m, g = [], []
    for s in seeds:
        r = ev(rule, B=4000, seed=s)
        m.append(r['mlp']['score']); g.append(r['gru']['score'])
    return np.mean(m), np.mean(g)


for S in [11, 14]:
    for mode, ns in [('total', [5, 99]), ('consec', [3, 4, 5]), ('decay', [0.6, 0.75, 0.85])]:
        for n in ns:
            m, g = both(var(13, 1.7, 2.2, S, mode, n))
            print(f'S={S} {mode:6s} {n:5} mlp {m:6.1f} gru {g:6.1f} mean {(m + g) / 2:6.1f}', flush=True)
