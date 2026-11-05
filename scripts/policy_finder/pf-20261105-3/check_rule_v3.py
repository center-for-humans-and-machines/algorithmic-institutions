"""v3: run the v3 rule on a hand-made history (c = 0, 4, 8, 12 / 15, 17, 20, timed out)."""
import torch as th
from aimanager.manager.rule import read_rule
_, code = read_rule('configs/managers/rule_based/pf-20261105-3-v3.yml', 'exp1')
for P in [dict(thr=20., pmax=30., gamma=2.09, horizon=9., nudge=0., mthr=17., stop=20.),   # ~v2 best
          dict(thr=21., pmax=35., gamma=2.0, horizon=8., nudge=2., mthr=18., stop=20.)]:
    print(P)
    for t in (0, 7, 8, 19, 20, 23):
        c = th.tensor([[0., 4., 8., 12.], [15., 17., 20., 0.]]).unsqueeze(-1).repeat(1, 1, t + 1)
        valid = th.ones_like(c); valid[1, 3, -1] = 0
        env = dict(c=c, valid=valid, p=th.zeros(2, 4, t), t=th.tensor(float(t)), th=th,
                   **{k: th.tensor(v) for k, v in P.items()})
        exec(code, {'__builtins__': {}}, env)
        print(' ', t, env['punishment'].clamp(0, 30).floor().tolist())
