"""Check v7: group 0 = players 0-3 contributing 0, 3, 6, 10 (others at 10). Prints group 0's punishment by
round for a few (w1, w2, later) settings (rel 0.85, slope 9, last_round 10)."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
base = dict(rel=0.85, slope=9.0, last_round=10)
for w1, w2, later in [(0.8, 0.1, 1.0), (0.8, 0.1, 1.5), (0.8, 0.0, 0.5), (1.5, 1.0, 2.5)]:
    code, params = load_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261104-9-v7.yml', dict(base, w1=w1, w2=w2, later=later))
    line = f'w1 {w1} w2 {w2} later {later}:'
    for t in [0, 1, 2, 3, 4, 5, 6, 8, 10, 11]:
        C = th.full((1, 8, t + 1), 10.); C[0, :4] = th.tensor([0., 3., 6., 10.]).view(4, 1)
        G = th.tensor([0, 0, 0, 0, 1, 1, 1, 1]).view(1, 8, 1).repeat(1, 1, t + 1)
        inp = rule_inputs(C, th.ones_like(C).bool(), th.zeros(1, 8, t), G, 0)
        out = run_rule(code, params, inp)[0].clamp(0, 30).floor()
        line += f' t{t}:{[int(x) for x in out[:3].tolist()]}'
    print(line)
