# v6 (k0=10, k1=3, kn=12, s0=0, nb=2): hand-made history. Players 0-3 in our group from round 0;
# player 4 joins at round 4, player 5 joins at round 12 (else in the other group); 6-7 never with us.
# Contributions 0,5,10,15 | 2 | 3 | 20s.
import torch as th
from aimanager.manager.rule import load_rule, run_rule
th.ones(2).sum(-1, keepdim=True); th.remainder(th.tensor(1.), 4); th.where(th.tensor(True), th.ones(1), th.ones(1)); th.clamp(th.ones(1), min=0)
th.flip(th.ones(1, 2), dims=(-1,)); th.cumprod(th.ones(1, 2), dim=-1); th.ones_like(th.ones(1))  # warm-up (lazy imports)
cfg = 'configs/managers/rule_based/pf-20261106-9-v6.yml'
code, P = load_rule(cfg, dict(k0=10.0, k1=3.0, kn=12.0, s0=0.0, nb=2), 'exp2_multi')
def run(t):
    c = th.tensor([[0, 5, 10, 15, 2, 3, 20, 20]], dtype=th.float).unsqueeze(-1).repeat(1, 1, t + 1)
    ing = th.zeros(1, 8, t + 1)
    ing[0, :4, :] = 1
    ing[0, 4, 4:] = 1
    ing[0, 5, 12:] = 1
    return run_rule(code, P, dict(c=c, valid=th.ones_like(c), p=th.zeros(1, 8, t), in_group=ing, t=th.tensor(float(t))))
for t in [0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 13]:
    print(t, [round(x, 2) for x in run(t)[0].tolist()])
