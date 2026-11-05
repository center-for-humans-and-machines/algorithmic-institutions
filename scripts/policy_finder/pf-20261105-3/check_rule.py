"""Run the rule on a hand-made history to check its output."""
import torch as th
from aimanager.manager.rule import read_rule
_, code = read_rule('configs/managers/rule_based/pf-20261105-3.yml', 'exp1')
P = dict(thr=14., slope=1., cap=8., horizon=16., taper=1.)
for t in (0, 8, 15, 16, 23):
    c = th.tensor([[0., 5., 12., 20.]]).unsqueeze(-1).repeat(1, 1, t + 1)
    valid = th.ones_like(c); valid[0, 1, -1] = 0
    env = dict(c=c, valid=valid, p=th.zeros(1, 4, t), t=th.tensor(float(t)), th=th,
               **{k: th.tensor(v) for k, v in P.items()})
    exec(code, {'__builtins__': {}}, env)
    print(t, env['punishment'].floor().tolist())
