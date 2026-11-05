# Sanity check of v2's rule on hand-made contributions (fresh process: catches the import trap)
import torch as th
from aimanager.manager.rule import run_rule, load_rule
code, params = load_rule('configs/managers/rule_based/pf-20261105-11-v2.yml', dict(T=16.0, A=1.0, B=1.5, S=8), 'exp1')
c = th.tensor([[0., 5, 10, 15.5], [16, 20, 0, 12]]).unsqueeze(-1).repeat(1, 1, 24)
for t in (0, 8, 9, 23):
    cc = c[..., :t + 1]
    inp = dict(c=cc, valid=th.ones_like(cc), p=cc[..., :t], t=th.tensor(float(t)))
    print(t, run_rule(code, params, inp))
