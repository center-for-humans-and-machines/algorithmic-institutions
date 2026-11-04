"""Sanity check of the v3 rule on a hand-made history, in a fresh process. Player 0 has received
5 points in every past round, the others nothing."""
import torch as th
from aimanager.manager.rule import run_rule
import yaml
path = 'configs/managers/rule_based/pf-20261104-5-v3.yml'
code = yaml.safe_load(open(path))['code']
params = dict(k=3.0, thr=8.0, cap=20.0, budget=20.0, quiet=0.5, t_stop=12)
for t in [0, 1, 3, 5, 12, 13, 23]:
    T = t + 1
    c = th.tensor([0, 0, 4, 7, 11, 0, 2, 10.]).view(1, 8, 1).repeat(1, 1, T)
    valid = th.ones_like(c); valid[0, 5] = 0
    ig = th.tensor([1, 1, 1, 1, 1, 1, 0, 0.]).view(1, 8, 1).repeat(1, 1, T)
    p = th.zeros(1, 8, t); p[0, 0] = 5.
    inp = dict(c=c, valid=valid, p=p, in_group=ig, t=th.tensor(float(t)))
    pr = {k_: th.tensor(float(v)) for k_, v in params.items()}
    print(t, run_rule(code, pr, inp))
