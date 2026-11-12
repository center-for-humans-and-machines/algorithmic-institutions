"""v6: toy-history check of the v6 rule and a summary of its Sobol design. Game 0: the rule's group holds players
0-6; game 1 players 0-3. Player 0 gives 0, 1 gives 3, 2 gives 5, 3 gives 8, 4 gives 2, the rest 18; in game 1
player 1 times out at the current round."""
import collections, torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule, read_rule, sobol_design
path = 'configs/managers/rule_based/pf-20261112-6-v6.yml'
code, prm = load_rule(path, dict(lo=1, hi=5, amt=21.7), setting='exp2_diff')
for t in [0, 3, 4, 5, 8, 12, 20, 23]:
    T = t + 1
    c = th.full((2, 8, T), 18.0); c[:, 0] = 0; c[:, 1] = 3; c[:, 2] = 5; c[:, 3] = 6; c[:, 4] = 1
    valid = th.ones(2, 8, T); valid[1, 1, -1] = 0; p = th.zeros(2, 8, t)
    ag = th.ones(2, 8, T, dtype=th.long); ag[0, :7] = 0; ag[1, :4] = 0
    out = run_rule(code, prm, rule_inputs(c, valid, p, ag, 0, setting='exp2_diff'))
    print(t, out[0].tolist(), out[1].tolist())
rule, _ = read_rule(path, 'exp2_diff')
D = sobol_design(rule, 256)
print(len(D), sorted(collections.Counter(d['lo'] for d in D).items()), sorted(collections.Counter(d['hi'] for d in D).items()))
print('amt', min(d['amt'] for d in D), max(d['amt'] for d in D), sum(d['amt'] >= 20 for d in D), sorted(collections.Counter(d['lo'] for d in D if d['amt'] >= 20).items()))
