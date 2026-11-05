import torch as th
from aimanager.manager.rule import load_rule, run_rule
th.ones(2).sum(-1, keepdim=True); th.remainder(th.tensor(1.), 4); th.minimum(th.ones(1), th.ones(1))  # warm-up
cfg = 'configs/managers/rule_based/pf-20261106-9-v2.yml'
code, P = load_rule(cfg, dict(k=5.0, s0=1.0, cap=20.0, quiet=2, t_end=16), 'exp2_multi')
def run(t):
    c = th.tensor([[0, 5, 10, 15, 20, 20, 20, 20]], dtype=th.float).unsqueeze(-1).repeat(1, 1, t + 1)
    ing = th.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=th.float).unsqueeze(-1).repeat(1, 1, t + 1)
    return run_rule(code, P, dict(c=c, valid=th.ones_like(c), p=th.zeros(1, 8, t), in_group=ing, t=th.tensor(float(t))))
for t in [0, 1, 2, 3, 4, 15, 16, 20]:
    print(t, run(t).tolist())
