"""Sanity check of the v6 rule (share guard) on hand-made histories, in a fresh process."""
import torch as th
from aimanager.manager.rule import run_rule
import yaml
path = 'configs/managers/rule_based/pf-20261104-5-v6.yml'
code = yaml.safe_load(open(path))['code']
ig1 = th.tensor([1, 1, 1, 1, 0, 0, 0, 0.])
for label, cc in [('ours low', [0, 4, 7, 11, 10, 10, 12, 8.]), ('ours high', [0, 15, 15, 11, 5, 5, 6, 8.]),
                  ('even', [0, 4, 12, 14, 0, 4, 12, 14.])]:
    for params in [dict(k=6.0, thr=10.0, w=0.5, t_stop=12, m=-100.0), dict(k=6.0, thr=10.0, w=0.5, t_stop=12, m=0.0),
                   dict(k=6.0, thr=10.0, w=0.5, t_stop=12, m=-5.0)]:
        print(label, params)
        for t in [0, 1, 2, 3, 4, 13]:
            T = t + 1
            c = th.tensor(cc).view(1, 8, 1).repeat(1, 1, T)
            valid = th.ones_like(c)
            ig = ig1.view(1, 8, 1).repeat(1, 1, T)
            p = th.zeros(1, 8, t); p[0, 4:, :] = 3.0   # ah punishes 3 each per round
            inp = dict(c=c, valid=valid, p=p, in_group=ig, t=th.tensor(float(t)))
            pr = {k_: th.tensor(float(v)) for k_, v in params.items()}
            print('  t', t, [round(x, 1) for x in run_rule(code, pr, inp)[0].tolist()])
