"""v4: run the v4 rule on a hand-made history in a fresh process: who is fined when.
Group: players 0-4 in my group (player 4 timed out this round), 5-7 in the other group."""
import torch as th
from aimanager.manager.rule import read_rule, run_rule
rule, code = read_rule('configs/managers/rule_based/pf-20261112-2-v4.yml')
params = dict(gap=10.0, fine=4.0, n_min=4, start=1, last_block=5)
contrib = th.tensor([2., 15., 16., 14., 0., 0., 20., 20.])
for t in [0, 3, 4, 11, 19, 20, 23]:
    T = t + 1
    c = contrib[None, :, None].repeat(1, 1, T)
    valid = th.ones_like(c); valid[0, 4, -1] = 0
    ing = th.zeros_like(c); ing[0, :5] = 1
    p = th.zeros(1, 8, t)
    out = run_rule(code, {k: float(v) for k, v in params.items()}, dict(c=c, valid=valid, p=p, in_group=ing, t=th.tensor(float(t))))
    print(t, out.tolist())
# small group: only players 0-2 in my group -> size 3 < n_min 4, no fine
ing = th.zeros(1, 8, 5); ing[0, :3] = 1
c = contrib[None, :, None].repeat(1, 1, 5)
print('size 3, t=4', run_rule(code, {k: float(v) for k, v in params.items()}, dict(c=c, valid=th.ones_like(c), p=th.zeros(1, 8, 4), in_group=ing, t=th.tensor(4.))).tolist())
