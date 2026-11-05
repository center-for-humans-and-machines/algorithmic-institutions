# Sanity check: the rule's punishments on hand-made contributions
import torch as th
from aimanager.manager.rule import run_rule, load_rule
code, params = load_rule('configs/managers/rule_based/pf-20261105-11.yml', dict(T=15.0, P=8.0, g=0.5, E=3), 'exp1')
c = th.tensor([[0.,5,10,14],[15,20,0,12]]).unsqueeze(-1).repeat(1,1,22)
for t in (0, 20, 21):
    cc = c[..., :t+1]
    inp = dict(c=cc, valid=th.ones_like(cc), p=cc[..., :t], t=th.tensor(float(t)))
    print(t, run_rule(code, params, inp))
