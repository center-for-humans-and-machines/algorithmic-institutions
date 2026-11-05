"""v2: run the v2 rule on a hand-made history (c = 0, 4, 8, 12, 15, 20 split over two games)."""
import torch as th
from aimanager.manager.rule import read_rule
_, code = read_rule('configs/managers/rule_based/pf-20261105-3-v2.yml', 'exp1')
for P in [dict(thr=15., pmax=30., gamma=1.5, warn=1., horizon=14., taper=0.5),
          dict(thr=15., pmax=29.35, gamma=1., warn=0., horizon=11., taper=0.29)]:  # ~v1 best
    print(P)
    for t in (0, 7, 13, 14, 23):
        c = th.tensor([[0., 4., 8., 12.], [15., 14., 20., 0.]]).unsqueeze(-1).repeat(1, 1, t + 1)
        valid = th.ones_like(c); valid[1, 3, -1] = 0
        env = dict(c=c, valid=valid, p=th.zeros(2, 4, t), t=th.tensor(float(t)), th=th,
                   **{k: th.tensor(v) for k, v in P.items()})
        exec(code, {'__builtins__': {}}, env)
        print(' ', t, env['punishment'].clamp(0, 30).floor().tolist())
