"""Sanity check of the rule on a hand-made history."""
import torch as th
from aimanager.manager.rule import load_rule, run_rule
import yaml
path = 'configs/managers/rule_based/pf-20261104-5.yml'
code = yaml.safe_load(open(path))['code']
params = dict(k=1.0, thr=8.0, cap=5.0, quiet=0.5, t_stop=17)
for t in [0, 3, 11, 17, 18]:
    T = t + 1
    c = th.tensor([0, 4, 7, 8, 20, 0, 2, 10.]).view(1, 8, 1).repeat(1, 1, T)
    valid = th.ones_like(c); valid[0, 5] = 0
    ig = th.tensor([1, 1, 1, 1, 1, 1, 0, 0.]).view(1, 8, 1).repeat(1, 1, T)
    inp = dict(c=c, valid=valid, p=th.zeros(1, 8, t), in_group=ig, t=th.tensor(float(t)))
    pr = {k_: float(v) for k_, v in params.items()}
    print(t, run_rule(code, pr, inp))
