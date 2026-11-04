"""Check v2's timing multipliers: one member at 0 among others at 10, by round."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
code, params = load_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261104-9-v2.yml',
                         dict(rel=1.0, slope=1.0, cap=30.0, fresh=2.0, pre_switch=0.5, last_round=21))
for t in [0, 1, 2, 3, 4, 7, 19, 20, 21, 22, 23]:
    C = th.full((1, 8, t + 1), 10.0); C[0, 0, :] = 0
    G = th.tensor([0, 0, 0, 0, 1, 1, 1, 1]).view(1, 8, 1).repeat(1, 1, t + 1)
    inp = rule_inputs(C, th.ones_like(C).bool(), th.zeros(1, 8, t), G, 0)
    print(t, run_rule(code, params, inp)[0].tolist())
