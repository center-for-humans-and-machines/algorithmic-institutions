"""Hand checks of v5's rule at round 2: player 0 punished in round 1, player 1 in round 0 only,
player 2 never, player 3 timed out; all contribute 4 now (player 3: 0, invalid); threshold 10, rate 2
-> raw 12. Also round 0 (empty p) and t >= t_end."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
R = 'configs/managers/rule_based/pf-20261104-7-v5.yml'
def run(params, c, valid, p, grp):
    code, prm = load_rule(R, params)
    return run_rule(code, prm, rule_inputs(c, valid, p, grp, 0))
c = th.tensor([[[5, 5, 4], [5, 5, 4], [5, 5, 4], [5, 5, 0]] + [[10, 10, 10]] * 4]).float()
valid = th.ones_like(c); valid[0, 3, 2] = 0
grp = th.tensor([[[0, 0, 0]] * 4 + [[1, 1, 1]] * 4])
p = th.zeros(1, 8, 2); p[0, 0, 1] = 5; p[0, 1, 0] = 5; p[0, 4, 1] = 3
base = {'threshold': 10.0, 'rate': 2.0, 't_end': 10}
print('cd1 rf0.5 (expect 6,12,12,0,0..)', run({**base, 'cooldown': 1, 'repeat_factor': 0.5}, c, valid, p, grp))
print('cd2 rf0   (expect 0,0,12,0,0..)', run({**base, 'cooldown': 2, 'repeat_factor': 0.0}, c, valid, p, grp))
print('cd3 rf1   (expect 12,12,12,0..)', run({**base, 'cooldown': 3, 'repeat_factor': 1.0}, c, valid, p, grp))
print('t_end 4 at round 5 (expect zeros)', run({**base, 't_end': 4, 'cooldown': 1, 'repeat_factor': 1.0}, c.repeat(1, 1, 2), valid.repeat(1, 1, 2), th.cat([p, p, p[..., :1]], -1), grp.repeat(1, 1, 2)))
print('round 0   (expect 10 x3 at c=5, 0)', run({**base, 'cooldown': 2, 'repeat_factor': 0.0}, c[..., :1], valid[..., :1], p[..., :0], grp[..., :1]))
