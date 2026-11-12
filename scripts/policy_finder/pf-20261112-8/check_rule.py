"""Which rounds does the rule punish in, for a fixed history (sanity check)."""
import torch as th
from aimanager.manager.rule import read_rule, run_rule, rule_inputs
rule, code = read_rule('configs/managers/rule_based/pf-20261112-8.yml', 'exp2_diff')
params = dict(thr=2, amount=2.5, last_pos=1, start=4)
for t in range(24):
    c = th.tensor([[0, 2, 3, 20, 0, 5, 10, 1]]).repeat(t + 1, 1).T[None]
    inp = rule_inputs(c, th.ones_like(c), th.zeros(1, 8, t), th.zeros_like(c), 0, 'exp2_diff')
    print(t, run_rule(code, params, inp)[0].tolist())
