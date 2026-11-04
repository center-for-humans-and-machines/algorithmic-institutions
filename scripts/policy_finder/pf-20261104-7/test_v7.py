"""Hand checks of v7's rule at round 13 (14 rounds of history), t_end 8, so only the norm clause.
My group (id 0): players 0-4. Player 4 joined at round 12 (was in group 1 before), player 3
joined at round 8. Contributions now: 12, 12, 12, 6, 6. Others' mean for player 3/4: (12*3+6)/4
= 10.5; slack 2 -> bar 8.5, shortfall 2.5; late_rate 4 -> 10. For players 0-2 others' mean 10.5
too (12,12,6,6 -> 9), bar 7 -> no shortfall."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
R = 'configs/managers/rule_based/pf-20261104-7-v7.yml'
T = 14
c = th.full((1, 8, T), 12.0); c[0, 3:5, -1] = 6.0
valid = th.ones_like(c)
grp = th.ones(1, 8, T); grp[0, :3] = 0; grp[0, 3, 8:] = 0; grp[0, 4, 12:] = 0
p = th.zeros(1, 8, T - 1)
def run(params, t=T):
    code, prm = load_rule(R, params)
    return run_rule(code, prm, rule_inputs(c[..., :t], valid[..., :t], p[..., :t - 1], grp[..., :t], 0))
base = {'threshold': 9.0, 'rate': 7.0, 't_end': 8, 'late_rate': 4.0, 'slack': 2.0}
print('window 1, inc 0 (player 4 arrived at 12, round 13 is its 2nd: expect zeros)\n', run({**base, 'window': 1, 'incumbent_factor': 0.0}))
print('window 2, inc 0 (expect player 4: 10, rest 0)\n', run({**base, 'window': 2, 'incumbent_factor': 0.0}))
print('window 4, inc 0 (expect player 4: 10; player 3 joined at 8, outside in round 7 not in 9..12 -> 0)\n', run({**base, 'window': 4, 'incumbent_factor': 0.0}))
print('window 1, inc 0.5 (neither is new: expect players 3, 4: 5)\n', run({**base, 'window': 1, 'incumbent_factor': 0.5}))
print('t_end 16 (core: 6 -> rate 7 * 3 = 21 for players 3, 4)\n', run({**base, 't_end': 16, 'window': 1, 'incumbent_factor': 0.0}))
print('round 0 (core only, everyone gives 12 -> zeros)\n', run({**base, 'window': 4, 'incumbent_factor': 1.0}, t=1))
