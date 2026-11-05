# v5 (k0=10, k1=4, w=0.1, s0=0, quiet=2, t_end=16): hand-made history. Players 0-3 in our group
# from round 0; player 4 joins us at round 4 (in the other group before); players 5-7 stay in the
# other group, whose manager punishes each of 4-7 by 8 points in round 0 only (q = 8 at round 1,
# 8/4 = 2 at round 4... over member-rounds). Contributions 0,5,10,15 | 2 (joiner) | 20s.
import torch as th
from aimanager.manager.rule import load_rule, run_rule
th.ones(2).sum(-1, keepdim=True); th.remainder(th.tensor(1.), 4); th.where(th.tensor(True), th.ones(1), th.ones(1)); th.clamp(th.ones(1), min=0)
th.flip(th.ones(1, 2), dims=(-1,)); th.cumprod(th.ones(1, 2), dim=-1); th.ones_like(th.ones(1))  # warm-up (lazy imports)
cfg = 'configs/managers/rule_based/pf-20261106-9-v5.yml'
code, P = load_rule(cfg, dict(k0=10.0, k1=4.0, w=0.1, s0=0.0, quiet=2, t_end=16), 'exp2_multi')
def run(t, harsh=True):
    c = th.tensor([[0, 5, 10, 15, 2, 20, 20, 20]], dtype=th.float).unsqueeze(-1).repeat(1, 1, t + 1)
    ing = th.zeros(1, 8, t + 1)
    ing[0, :4, :] = 1
    ing[0, 4, 4:] = 1
    p = th.zeros(1, 8, t)
    if harsh and t > 0:
        p[0, 4:, 0] = 8
    return run_rule(code, P, dict(c=c, valid=th.ones_like(c), p=p, in_group=ing, t=th.tensor(float(t))))
for t in [0, 1, 2, 3, 4, 5, 6, 8, 9, 16]:
    print(t, 'harsh other', [round(x, 2) for x in run(t)[0].tolist()], '| lenient other', [round(x, 2) for x in run(t, False)[0].tolist()])
