"""Toy check: who the rule punishes, and when."""
import torch as th
import yaml
from aimanager.manager.rule import run_rule
cfg = yaml.safe_load(open('configs/managers/rule_based/pf-20261104-4.yml'))
params = {'thr': 5, 'amount': 2, 'm': 2, 't_stop': 20}
for t in [0, 1, 2, 3, 4, 19, 20, 21, 23]:
    c = th.tensor([0., 4, 5, 20, 0, 3, 10, 20]).view(1, 8, 1).repeat(1, 1, t + 1)
    valid = th.ones_like(c); valid[0, 1, -1] = 0
    ig = th.tensor([1., 1, 1, 1, 0, 0, 0, 0]).view(1, 8, 1).repeat(1, 1, t + 1)
    p = th.zeros(1, 8, t)
    out = run_rule(cfg['code'], {k: th.tensor(float(v)) for k, v in params.items()},
                   dict(c=c, valid=valid, p=p, in_group=ig, t=th.tensor(float(t))))
    print(t, out.tolist())
