# v7 (k0=10, k1=3, s0=0, r=0.5): hand-made history. Players 0-3 in our group from round 0;
# player 4 joins at round 4, player 5 joins at round 8, player 7 at round 12 (else other group);
# player 6 never with us. Contributions 0,5,10,15 | 2 | 3 | 20 | 1. Past punishment: what this
# rule itself would have given (simulated round by round), plus 10 from the other manager to
# player 4 in round 3 (before it joined: must not count as a repeat).
import torch as th
from aimanager.manager.rule import load_rule, run_rule
th.ones(2).sum(-1, keepdim=True); th.remainder(th.tensor(1.), 4); th.where(th.tensor(True), th.ones(1), th.ones(1)); th.clamp(th.ones(1), min=0)
th.flip(th.ones(1, 2), dims=(-1,)); th.cumprod(th.ones(1, 2), dim=-1); th.ones_like(th.ones(1))
cfg = 'configs/managers/rule_based/pf-20261106-9-v7.yml'
code, P = load_rule(cfg, dict(k0=10.0, k1=3.0, s0=0.0, r=0.5), 'exp2_multi')
cv = th.tensor([0, 5, 10, 15, 2, 3, 20, 1], dtype=th.float)
ing = th.zeros(1, 8, 24); ing[0, :4] = 1; ing[0, 4, 4:] = 1; ing[0, 5, 8:] = 1; ing[0, 7, 12:] = 1
past = th.zeros(1, 8, 24); past[0, 4, 3] = 10
for t in range(14):
    c = cv.view(1, 8, 1).repeat(1, 1, t + 1)
    out = run_rule(code, P, dict(c=c, valid=th.ones_like(c), p=past[..., :t].clone(), in_group=ing[..., :t + 1], t=th.tensor(float(t))))
    pun = th.clamp(out, 0, 30).floor() * ing[..., t]
    past[..., t] = pun
    print(t, [round(x, 2) for x in out[0].tolist()])
