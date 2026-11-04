"""Sanity check of the v5 rule on a hand-made history, in a fresh process."""
import torch as th
from aimanager.manager.rule import run_rule
import yaml
path = 'configs/managers/rule_based/pf-20261104-5-v5.yml'
code = yaml.safe_load(open(path))['code']
for params in [dict(k=3.0, thr=8.0, w=0.5, d=0.5, t_stop=12), dict(k=10.0, thr=10.0, w=0.0, d=0.6, t_stop=23)]:
    print(params)
    for t in [0, 1, 3, 4, 5, 8, 12, 13, 16, 23]:
        T = t + 1
        c = th.tensor([0, 0, 4, 7, 11, 0, 2, 10.]).view(1, 8, 1).repeat(1, 1, T)
        valid = th.ones_like(c); valid[0, 5] = 0
        ig = th.tensor([1, 1, 1, 1, 1, 1, 0, 0.]).view(1, 8, 1).repeat(1, 1, T)
        p = th.zeros(1, 8, t)
        inp = dict(c=c, valid=valid, p=p, in_group=ig, t=th.tensor(float(t)))
        pr = {k_: th.tensor(float(v)) for k_, v in params.items()}
        print(' ', t, [round(x, 2) for x in run_rule(code, pr, inp)[0].tolist()])
