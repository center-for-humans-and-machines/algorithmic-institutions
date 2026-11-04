"""Sanity check of the v2 rule on a hand-made history, in a fresh process."""
import torch as th
from aimanager.manager.rule import run_rule
import yaml
path = 'configs/managers/rule_based/pf-20261104-5-v2.yml'
code = yaml.safe_load(open(path))['code']
params = dict(k=2.0, thr=12.0, cap=15.0, decay=0.9, t_stop=9)
for t in [0, 1, 5, 9, 10, 23]:
    T = t + 1
    c = th.tensor([0, 4, 7, 11, 20, 0, 2, 10.]).view(1, 8, 1).repeat(1, 1, T)
    valid = th.ones_like(c); valid[0, 5] = 0
    ig = th.tensor([1, 1, 1, 1, 1, 1, 0, 0.]).view(1, 8, 1).repeat(1, 1, T)
    inp = dict(c=c, valid=valid, p=th.zeros(1, 8, t), in_group=ig, t=th.tensor(float(t)))
    pr = {k_: th.tensor(float(v)) for k_, v in params.items()}
    print(t, run_rule(code, pr, inp))
