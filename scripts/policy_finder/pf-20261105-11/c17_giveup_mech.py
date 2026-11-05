"""v3: how does give-up change the dynamics? Per-round c and p on the MLP surrogate, and long
windows (S up to 23) with give-up."""
import sys, numpy as np; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from screen import *
from c16_giveup_check import capped
np.set_printoptions(linewidth=200)
for name, a in [('T13 B2.2 S11', (13, 1.7, 2.2, 11)), ('T13 B2.2 S11 n5', (13, 1.7, 2.2, 11, 5))]:
    x = SIMS['gru'].run(capped(*a), B=8000, seed=3, return_traj=True)
    print(name, round(x['score'], 1))
    print('  c', x['c_round'].round(1)); print('  p', x['p_round'].round(2))
    c, p, v = x['c'], x['p'], x['v']
    npun = (p[..., :12] > 0).sum(-1)
    late = (c[..., 18:] * v[..., 18:]).sum(-1) / v[..., 18:].sum(-1).clamp(min=1)
    for k in [0, 1, 2, 3, 4, 5, 6, 8, 12]:
        mm = npun == k
        if mm.sum() > 50:
            print(f'   punished {k} rounds: share {mm.float().mean():.3f} c11 {c[..., 11][mm].mean():.1f} late {late[mm].mean():.1f}')
for S in [14, 17, 20, 23]:
    for n in [5, 6, 8]:
        r = ev(capped(13, 1.7, 2.2, S, n), B=4000, seed=11)
        show(f'T13 B2.2 S{S} n{n}', r)
