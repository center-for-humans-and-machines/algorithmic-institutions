"""Hand checks of v6's rule at round 10 (11 rounds of history). Game 0: my group (group id 0) has
players 0-5 (6 members); game 1: players 0-1 only (2 members). Everyone gives 4 (player 1 timed out
in game 0 -> 0, invalid). threshold 10, rate 2 -> raw 12."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
R = 'configs/managers/rule_based/pf-20261104-7-v6.yml'
T = 11
c = th.full((2, 8, T), 4.0)
valid = th.ones_like(c); valid[0, 1, -1] = 0; c[0, 1, -1] = 0
grp = th.ones(2, 8, T); grp[0, :6] = 0; grp[1, :2] = 0  # group ids; mine is 0
p = th.zeros(2, 8, T - 1)
def run(params, t=T):
    code, prm = load_rule(R, params)
    return run_rule(code, prm, rule_inputs(c[..., :t], valid[..., :t], p[..., :t - 1], grp[..., :t], 0))
base = {'threshold': 10.0, 'rate': 2.0}
print('t_end 8, big 12, n_big 5 (expect game0: 12,0,12,12,12,12,0,0 ; game1: zeros)\n', run({**base, 't_end': 8, 't_end_big': 12, 'n_big': 5}))
print('t_end 12, big 8, n_big 5 (expect game0 zeros; game1: 12,12,0..)\n', run({**base, 't_end': 12, 't_end_big': 8, 'n_big': 5}))
print('n_big 7 (game0 not big): t_end 8, big 12 (expect zeros)\n', run({**base, 't_end': 8, 't_end_big': 12, 'n_big': 7}))
print('round 0, t_end 4, big 6 (expect 12 for my valid members)\n', run({**base, 't_end': 4, 't_end_big': 6, 'n_big': 8}, t=1))
