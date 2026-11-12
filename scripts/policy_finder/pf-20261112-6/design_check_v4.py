"""v4: toy-history check of the v4 rule and a summary of its Sobol design. Game 0: the rule's group holds players
0-6; game 1 players 0-3. Contributions: player 0 gives 0, 1 gives 3, 2 gives 8, 3 gives 14, the rest 18."""
import collections, torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule, read_rule, sobol_design
path = 'configs/managers/rule_based/pf-20261112-6-v4.yml'
code, prm = load_rule(path, dict(last_b=2, thr=10, amt=2.5, lo=2, amt_fr=9.0), setting='exp2_diff')
for t in [0, 3, 4, 5, 8, 12, 20, 23]:
    T = t + 1
    c = th.full((2, 8, T), 18.0); c[:, 0] = 0; c[:, 1] = 3; c[:, 2] = 8; c[:, 3] = 14
    valid = th.ones(2, 8, T); valid[1, 1, -1] = 0; p = th.zeros(2, 8, t)
    ag = th.ones(2, 8, T, dtype=th.long); ag[0, :7] = 0; ag[1, :4] = 0
    out = run_rule(code, prm, rule_inputs(c, valid, p, ag, 0, setting='exp2_diff'))
    print(t, out[0].tolist(), out[1].tolist())
rule, _ = read_rule(path, 'exp2_diff')
D = sobol_design(rule, 256)
print(len(D), collections.Counter(d['last_b'] for d in D), sorted(collections.Counter(d['lo'] for d in D).items()))
print('thr', sorted(collections.Counter(d['thr'] for d in D).items()))
print('amt<2', sum(d['amt'] < 2 for d in D), 'amt_fr<1', sum(d['amt_fr'] < 1 for d in D), 'thr<=lo', sum(d['thr'] <= d['lo'] for d in D))
