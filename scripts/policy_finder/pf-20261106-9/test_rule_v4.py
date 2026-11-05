# v4 (n_stages=2, decay=0.5): hand-made history. Players 0-3 in our group from round 0; player 4 joins us at round 8
# (in the other group before); player 3 leaves at round 8. Contributions 0,5,10,15 | 2 (joiner).
import torch as th
from aimanager.manager.rule import load_rule, run_rule
th.ones(2).sum(-1, keepdim=True); th.remainder(th.tensor(1.), 4); th.minimum(th.ones(1), th.ones(1))
th.flip(th.ones(1, 2), dims=(-1,)); th.cumprod(th.ones(1, 2), dim=-1); th.pow(th.ones(1), th.ones(1)); th.floor(th.ones(1))  # warm-up
cfg = 'configs/managers/rule_based/pf-20261106-9-v4.yml'
code, P = load_rule(cfg, dict(k=5.0, s0=0.0, cap=20.0, n_stages=2, decay=0.5, quiet=2, t_end=20), 'exp2_multi')
def run(t):
    c = th.tensor([[0, 5, 10, 15, 2, 20, 20, 20]], dtype=th.float).unsqueeze(-1).repeat(1, 1, t + 1)
    ing = th.zeros(1, 8, t + 1)
    ing[0, :3, :] = 1
    ing[0, 3, :8] = 1
    ing[0, 4, 8:] = 1
    return run_rule(code, P, dict(c=c, valid=th.ones_like(c), p=th.zeros(1, 8, t), in_group=ing, t=th.tensor(float(t))))
for t in [0, 1, 2, 4, 5, 8, 9, 10, 12, 13, 19, 20]:
    print(t, run(t).tolist())
