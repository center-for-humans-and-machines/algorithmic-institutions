"""Hand checks of v4's rule: cap off (margin -1000) equals the plain threshold rule; cap binding
scales punishments so that my post-punishment cg = other cg + margin; round-0 other estimate."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
R = 'configs/managers/rule_based/pf-20261104-7-v4.yml'
def run(params, c, valid, p, grp):
    code, prm = load_rule(R, params)
    return run_rule(code, prm, rule_inputs(c, valid, p, grp, 0))
# round 1: my group players 0-3, other 4-7
c = th.tensor([[[10, 8], [10, 4], [10, 12], [10, 0], [10, 10], [10, 10], [10, 10], [10, 10]]]).float()
valid = th.ones_like(c); valid[0, 3, 1] = 0
grp = th.tensor([[[0, 0]] * 4 + [[1, 1]] * 4])
p = th.tensor([[[0], [0], [0], [0], [2], [2], [2], [2]]]).float()
base = {'threshold': 10.0, 'rate': 5.0, 't_end': 10}
off = run({**base, 'margin': -1000.0}, c, valid, p, grp)
print('cap off', off)  # expect [10, 30, 0, 0(timed out), 0...]
on = run({**base, 'margin': 0.0}, c, valid, p, grp)
pre = 1.6 * (8 + 4 + 12) / 3; oth = 1.6 * 10 - 2
print('cap 0', on, 'post cg', pre - on.sum() / 3, 'other', oth)
print('cap -5 (expect 2.85, 8.55)', run({**base, 'margin': -5.0}, c, valid, p, grp))
c0 = c[..., :1]; v0 = valid[..., :1]; g0 = grp[..., :1]
print('round 0', run({**base, 'margin': -5.0}, c0, v0, p[..., :0], g0))
