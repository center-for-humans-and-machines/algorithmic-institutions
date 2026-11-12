"""v7: toy-history check of the v7 rule and a summary of its Sobol design. Game 0: the rule's group holds players
0-6; game 1 players 0-3. Givers: player 0 gives 0, 1 gives 3, 2 gives 5, 3 gives 6, 4 gives 1, 5 gives 2, the rest
18; in game 1 player 1 times out at the current round."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule, read_rule, sobol_design
path = 'configs/managers/rule_based/pf-20261112-6-v7.yml'
code, prm = load_rule(path, dict(lo=1, mid=2, hi=5, amt_fr=29.3, amt=17.6), setting='exp2_diff')
for t in [0, 3, 4, 5, 8, 12, 20, 23]:
    T = t + 1
    c = th.full((2, 8, T), 18.0); c[:, 0] = 0; c[:, 1] = 3; c[:, 2] = 5; c[:, 3] = 6; c[:, 4] = 1; c[:, 5] = 2
    valid = th.ones(2, 8, T); valid[1, 1, -1] = 0; p = th.zeros(2, 8, t)
    ag = th.ones(2, 8, T, dtype=th.long); ag[0, :7] = 0; ag[1, :4] = 0
    out = run_rule(code, prm, rule_inputs(c, valid, p, ag, 0, setting='exp2_diff'))
    print(t, out[0].tolist(), out[1].tolist())
rule, _ = read_rule(path, 'exp2_diff')
D = sobol_design(rule, 256)
import numpy as np
A = np.array([[d['amt_fr'], d['amt']] for d in D])
print(len(D), 'amt_fr', A[:, 0].min().round(2), A[:, 0].max().round(2), 'amt', A[:, 1].min().round(2), A[:, 1].max().round(2),
      'amt_fr>=28', (A[:, 0] >= 28).sum(), 'amt 15-21', ((A[:, 1] >= 15) & (A[:, 1] <= 21)).sum(),
      'fixed', set((d['lo'], d['mid'], d['hi']) for d in D))
