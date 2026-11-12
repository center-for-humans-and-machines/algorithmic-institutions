"""v3: toy-history check of the v3 rule and a summary of its Sobol design. Two games: in game 0 the rule's group
holds players 0-6 (7 members), in game 1 players 0-3 (4 members). Player 0 always gives 0, player 1 gives 0 only
in the current round, player 2 gives 3 always, the rest give 12."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule, read_rule, sobol_design
path = 'configs/managers/rule_based/pf-20261112-6-v3.yml'
params = dict(n_min=7, last_b=4, thr=4, streak=2, amt=6.0)
code, prm = load_rule(path, params, setting='exp2_diff')
for t in [0, 3, 4, 5, 8, 16, 20, 23]:
    T = t + 1
    c = th.full((2, 8, T), 12.0); c[:, 0] = 0; c[:, 1, -1] = 0; c[:, 2] = 3
    valid = th.ones(2, 8, T); p = th.zeros(2, 8, t)
    ag = th.ones(2, 8, T, dtype=th.long); ag[0, :7] = 0; ag[1, :4] = 0
    out = run_rule(code, prm, rule_inputs(c, valid, p, ag, 0, setting='exp2_diff'))
    print(t, out[0].tolist(), out[1].tolist())
rule, _ = read_rule(path, 'exp2_diff')
D = sobol_design(rule, 256)
import collections
pts = D if isinstance(D, list) else D
print(type(D), len(D))
print(collections.Counter(d['n_min'] for d in D), collections.Counter(d['last_b'] for d in D), collections.Counter(d['streak'] for d in D))
print('amt<4', sum(d['amt'] < 4 for d in D), 'thr', sorted(collections.Counter(d['thr'] for d in D).items()))
