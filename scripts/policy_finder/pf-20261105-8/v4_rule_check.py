"""v4: run the v4 rule on a toy group (c = 0, 4, 10, 15, 18, 20) round by round at v3's best
mapped to v4's params (rate = 2.72/24) with a lower opening, and compare with v3's rule."""
import torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
v4 = 'configs/managers/rule_based/pf-20261105-8-v4.yml'
v3 = 'configs/managers/rule_based/pf-20261105-8-v3.yml'
p3 = dict(standard=24.0, dose=31.0, curvature=2.72, horizon=5, floor=1.0, tail=12)
p4 = dict(standard=24.0, rate=2.72 / 24, dose=31.0, opening=12.0, horizon=5, floor=1.0, end=17)
cc = th.tensor([0., 4, 10, 15, 18, 20])
for t in [0, 1, 4, 5, 16, 17, 23]:
    C = cc.view(1, 6, 1).repeat(1, 1, t + 1); V = th.ones_like(C); P = th.zeros(1, 6, t)
    out = []
    for path, p in [(v3, p3), (v4, p4)]:
        code, prm = load_rule(path, p, 'exp1')
        out.append(run_rule(code, prm, rule_inputs(C, V, P, th.zeros_like(C), 0, 'exp1')).clamp(0, 30).floor()[0].tolist())
    print('t', t, 'v3', out[0], 'v4', out[1])
