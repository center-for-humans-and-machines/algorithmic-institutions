"""Sanity check of the v7 rule (size elasticity) on hand-made histories."""
import torch as th, yaml
from aimanager.manager.rule import run_rule
code = yaml.safe_load(open('configs/managers/rule_based/pf-20261104-5-v7.yml'))['code']
for label, ig1 in [('n=4', [1, 1, 1, 1, 0, 0, 0, 0.]), ('n=2', [1, 1, 0, 0, 0, 0, 0, 0.]),
                   ('n=8', [1.] * 8)]:
    for params in [dict(k=6.0, thr=10.0, w=0.5, t_stop=12, s=0.0), dict(k=6.0, thr=10.0, w=0.5, t_stop=12, s=1.5)]:
        print(label, params)
        for t in [0, 3, 4, 5, 13]:
            T = t + 1
            c = th.tensor([0, 4, 8, 11, 3, 9, 12, 2.]).view(1, 8, 1).repeat(1, 1, T)
            ig = th.tensor(ig1).view(1, 8, 1).repeat(1, 1, T)
            inp = dict(c=c, valid=th.ones_like(c), p=th.zeros(1, 8, t), in_group=ig, t=th.tensor(float(t)))
            pr = {k_: th.tensor(float(v)) for k_, v in params.items()}
            print('  t', t, [round(x, 1) for x in run_rule(code, pr, inp)[0].tolist()])
