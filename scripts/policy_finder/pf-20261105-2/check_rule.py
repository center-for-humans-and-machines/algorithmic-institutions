"""Sanity check of the rule on hand-made histories."""
import torch as th
from aimanager.manager.rule import read_rule, run_rule
rule, code = read_rule('configs/managers/rule_based/pf-20261105-2.yml', 'exp1')
params = dict(c_star=12.0, k=0.6, p_max=6.0, H=10)
for t in [0, 10, 13, 18, 23]:
    c = th.tensor([[[0.], [8.], [12.], [20.]]]).repeat(1, 1, t + 1)
    valid = th.ones_like(c); valid[0, 0, -1] = 1
    p = th.zeros(1, 4, t)
    out = run_rule(code, params, dict(c=c, valid=valid, p=p, t=th.tensor(float(t))))
    print(t, out.tolist())
