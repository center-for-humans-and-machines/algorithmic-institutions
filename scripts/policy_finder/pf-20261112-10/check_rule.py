"""Sanity check of the rule's timing and targeting on a hand-made history."""
import torch as th
from aimanager.manager.rule import read_rule, run_rule
rule, code = read_rule('configs/managers/rule_based/pf-20261112-10.yml', 'exp2_diff')
for t, quiet, t_end in [(0, 1, 23), (3, 1, 23), (2, 2, 23), (1, 2, 23), (20, 3, 23), (21, 0, 20), (19, 0, 23)]:
    c = th.tensor([0, 3, 8, 20, 0, 0, 0, 0.]).view(1, 8, 1).repeat(1, 1, t + 1)
    ing = th.tensor([1, 1, 1, 1, 0, 0, 0, 0.]).view(1, 8, 1).repeat(1, 1, t + 1)
    valid = th.ones_like(c); valid[0, 1, -1] = 0
    inp = dict(c=c, valid=valid, p=th.zeros(1, 8, t), in_group=ing, t=th.tensor(float(t)))
    print(t, quiet, t_end, run_rule(code, dict(thr=5, amt=2, quiet=quiet, t_end=t_end), inp).tolist())
