"""Check v4: group 0 = players 0-3 contributing 0, 3, 6, 10; group 1 at 10. Prints group 0's
punishment by round (rel 0.9, top 60, power 1.5, mid 0.4, last_round 9), and a v3-like
setting (power 1, top 80) for comparison."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
for pr in [dict(rel=0.9, top=60.0, power=1.5, mid=0.4, last_round=9), dict(rel=0.84, top=80.0, power=1.0, mid=0.37, last_round=10)]:
    code, params = load_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261104-9-v4.yml', pr)
    print(pr)
    for t in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 23]:
        C = th.full((1, 8, t + 1), 10.0); C[0, 0, :] = 0; C[0, 1, :] = 3; C[0, 2, :] = 6
        G = th.tensor([0, 0, 0, 0, 1, 1, 1, 1]).view(1, 8, 1).repeat(1, 1, t + 1)
        inp = rule_inputs(C, th.ones_like(C).bool(), th.zeros(1, 8, t), G, 0)
        out = run_rule(code, params, inp)[0].clamp(0, 30).floor()
        print(' ', t, out.tolist())
