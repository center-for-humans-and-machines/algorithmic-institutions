# v3: cold-process check of the v3 rule on a hand case, and equivalence with v2 at decay=1, mid=1
import torch as th, yaml
from aimanager.manager.rule import run_rule
R = '/opt/algorithmic-institutions/configs/managers/rule_based/'
v3 = yaml.safe_load(open(R+'pf-20261106-11-v3.yml')); v2 = yaml.safe_load(open(R+'pf-20261106-11-v2.yml'))
c3 = compile(v3['code'], 'v3', 'exec'); c2 = compile(v2['code'], 'v2', 'exec')
def inp(t, g=None):
    th.manual_seed(t)
    c = th.randint(0, 21, (64, 8, t+1)).float()
    ig = (th.rand(64, 8, t+1) > 0.5).float() if g is None else g.repeat(1, 1, t+1)
    return dict(c=c, valid=(th.rand(64, 8, t+1) > 0.05).float(), p=th.randint(0, 31, (64, 8, t)).float(), in_group=ig, t=th.tensor(float(t)))
for t in range(24):
    x = inp(t)
    a = run_rule(c3, dict(rate=6.9, delta=-2.0, decay=1.0, mid=1.0), x)
    b = run_rule(c2, dict(rate=6.9, delta=-2.0, floor=0.0, late=0.0, kappa=0.0), x)
    assert th.allclose(a, b), t
print('v3(decay=1, mid=1) == v2(floor=0, late=0, kappa=0) on all rounds')
g = th.tensor([[1., 1, 1, 1, 0, 0, 0, 0]]).unsqueeze(-1)
for t in [0, 1, 2, 3, 4, 5, 8, 16, 17, 20, 21, 22, 23]:
    x = inp(t, g); x['c'][..., -1] = th.tensor([0., 10, 10, 20, 0, 0, 0, 0]); x['valid'] = th.ones_like(x['c'])
    print(t, run_rule(c3, dict(rate=2.0, delta=-2.0, decay=0.5, mid=0.5), x)[0, :4].tolist())
