"""Check v6: group 0 = players 0-3 contributing 0, 3, 6, 10 (others at 10). Prints group 0's punishment by
round for several decays (rel 0.85, slope 9, mid 0.4, last_round 16)."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
base = dict(rel=0.85, slope=9.0, mid=0.4, last_round=16)
for decay in (0.25, 0.6, 1.0, 1.6):
    code, params = load_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261104-9-v6.yml', dict(base, decay=decay))
    line = f'decay {decay:4.2f}:'
    for t in [0, 1, 3, 4, 5, 8, 9, 12, 16, 17]:
        C = th.full((1, 8, t + 1), 10.); C[0, :4] = th.tensor([0., 3., 6., 10.]).view(4, 1)
        G = th.tensor([0, 0, 0, 0, 1, 1, 1, 1]).view(1, 8, 1).repeat(1, 1, t + 1)
        inp = rule_inputs(C, th.ones_like(C).bool(), th.zeros(1, 8, t), G, 0)
        out = run_rule(code, params, inp)[0].clamp(0, 30).floor()
        line += f' t{t}:{[int(x) for x in out[:4].tolist()]}'
    print(line)
