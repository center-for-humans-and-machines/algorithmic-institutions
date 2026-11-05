"""v3: re-check the give-up candidates on fresh seeds with more games; compare a count cap
(stop after n punished rounds) with a points budget (stop once a player was charged K in total)."""
import sys, numpy as np; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from screen import *


def capped(T, A, B, S, n=99, K=1e9):
    def f(c, v, p, t):
        if t > S: return zeros(c)
        s = th.relu(T - c[..., -1])
        x = th.where(s > 0, A + B * s, zeros(c))
        if t == 0: return x
        k = (p > 0).float().sum(-1)
        tot = p.sum(-1)
        x = th.where(k >= n, zeros(c), x)
        return th.minimum(x, th.relu(K - tot))
    return f


cands = [('v2 best', (14.3, 1.7, 1.56, 8)), ('T13 B2.2 S8', (13, 1.7, 2.2, 8)),
         ('T13 B2.2 S11 n5', (13, 1.7, 2.2, 11, 5)), ('T13 B2.2 S11 n4', (13, 1.7, 2.2, 11, 4)),
         ('T13 B2.2 S11 n6', (13, 1.7, 2.2, 11, 6)), ('T13 B2.2 S14 n5', (13, 1.7, 2.2, 14, 5)),
         ('T14.3 B1.56 S11 n5', (14.3, 1.7, 1.56, 11, 5)), ('T13 B2.2 S11', (13, 1.7, 2.2, 11))]
if __name__ == "__main__":
  for K in [20, 30, 45, 60]:
      for S in [8, 11]:
          cands.append((f'T13 B2.2 S{S} K{K}', (13, 1.7, 2.2, S, 99, K)))
  for name, a in cands:
      r = {k: [] for k in ['mlp', 'gru']}
      for seed in [11, 12]:
          x = ev(capped(*a), B=4000, seed=seed)
          for k in r: r[k].append(x[k]['score'])
      m = {k: np.mean(v) for k, v in r.items()}
      print(f'{name:28s} mlp {m["mlp"]:6.1f} gru {m["gru"]:6.1f} mean {(m["mlp"] + m["gru"]) / 2:6.1f}', flush=True)
