"""Check v5: group 0 = players 0-3 contributing 0, 3, 6, 10; group 1 at a level L with last-round
punishment q per member. Prints group 0's punishment for several margins and other-group states."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
base = dict(rel=0.85, slope=9.0, mid=0.4, last_round=10)
for margin in (-20.0, -6.0, 0.0):
    for L, q in ((10.0, 0.0), (6.0, 0.0), (14.0, 0.0), (10.0, 5.0)):
        pr = dict(base, margin=margin)
        code, params = load_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261104-9-v5.yml', pr)
        line = f'margin {margin:5.1f} other c {L:4.1f} p_last {q}: '
        for t in [0, 1, 3, 4, 12]:
            C = th.full((1, 8, t + 1), L); C[0, :4] = th.tensor([0., 3., 6., 10.]).view(4, 1)
            G = th.tensor([0, 0, 0, 0, 1, 1, 1, 1]).view(1, 8, 1).repeat(1, 1, t + 1)
            Pn = th.zeros(1, 8, t); Pn[0, 4:] = q
            inp = rule_inputs(C, th.ones_like(C).bool(), Pn, G, 0)
            out = run_rule(code, params, inp)[0].clamp(0, 30).floor()
            line += f' t{t}:{[int(x) for x in out[:4].tolist()]}'
        print(line)
