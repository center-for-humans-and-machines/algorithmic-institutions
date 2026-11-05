# v5: v5 == v4 at kappa=0; stance multiplier on hand cases (other group punished 0 / 6 / 12 per member in round 0)
import torch as th, yaml
from aimanager.manager.rule import run_rule
R = '/opt/algorithmic-institutions/configs/managers/rule_based/'
v5 = compile(yaml.safe_load(open(R+'pf-20261106-11-v5.yml'))['code'], 'v5', 'exec')
v4 = compile(yaml.safe_load(open(R+'pf-20261106-11-v4.yml'))['code'], 'v4', 'exec')
g = th.tensor([[1., 1, 1, 1, 0, 0, 0, 0]]).unsqueeze(-1)
for t in range(24):
    th.manual_seed(t)
    x = dict(c=th.randint(0, 21, (64, 8, t+1)).float(), valid=th.ones(64, 8, t+1), p=th.randint(0, 31, (64, 8, t)).float(), in_group=g.repeat(64, 1, t+1), t=th.tensor(float(t)))
    pr = dict(rate=7.7, delta=-1.4, decay=0.5, mid=1.0)
    assert th.allclose(run_rule(v5, dict(pr, kappa=0.0), x), run_rule(v4, pr, x))
    x['c'][..., -1] = th.tensor([0., 10, 10, 20, 0, 0, 0, 0])
    x['p'][...] = 0
    out = []
    for q in [0., 6., 12.]:
        if t > 0: x['p'][:, 4:, 0] = q
        out.append(run_rule(v5, dict(pr, rate=1.0, delta=0.0, kappa=1.0), x)[0, 0].item())
    if t in (0, 1, 4, 5): print(t, [round(o, 3) for o in out])
print('v5 == v4 at kappa 0: ok')
