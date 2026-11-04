"""Check v3's timing: player 0 (incumbent) and player 4 (joins group 0 at round 12,
was in group 1 before) both contribute 0 among others at 10, by round."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
code, params = load_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261104-9-v3.yml',
                         dict(rel=1.0, slope=1.0, cap=10.0, fresh=2.0, last_round=6, newcomer=0.5))
for t in [0, 1, 3, 4, 6, 7, 8, 11, 12, 13, 16, 20]:
    C = th.full((1, 8, t + 1), 10.0); C[0, 0, :] = 0; C[0, 4, :] = 0
    G = th.tensor([0, 0, 0, 0, 1, 1, 1, 1]).view(1, 8, 1).repeat(1, 1, t + 1)
    if t >= 12: G[0, 4, 12:] = 0
    inp = rule_inputs(C, th.ones_like(C).bool(), th.zeros(1, 8, t), G, 0)
    print(t, [round(x, 2) for x in run_rule(code, params, inp)[0].tolist()])
