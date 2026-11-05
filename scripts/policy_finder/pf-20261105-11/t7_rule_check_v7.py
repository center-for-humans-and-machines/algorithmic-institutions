# v7: sanity check of the rule on hand-made contributions (fresh process); F 2.5 -> fines c <= 2 late, E 15.5 -> rounds 9..15
import torch as th
from aimanager.manager.rule import run_rule, load_rule
code, params = load_rule('configs/managers/rule_based/pf-20261105-11-v7.yml',
                         dict(T=13.28734753818213, A=2.4262768157156414, B=1.7607020146886698, S=8, F=2.5, E=15.5, L=30.0), 'exp1')
c = th.tensor([[0., 2, 3, 13], [16, 20, 0, 12]]).unsqueeze(-1).repeat(1, 1, 24)
v = th.ones_like(c); v[1, 2] = 0  # player 2 of game 1 timed out (c 0)
for t in (0, 8, 9, 15, 16, 23):
    cc = c[..., :t + 1]
    inp = dict(c=cc, valid=v[..., :t + 1], p=cc[..., :t], t=th.tensor(float(t)))
    print(t, run_rule(code, params, inp))
