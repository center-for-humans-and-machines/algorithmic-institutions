# v7: v7 == v3 at mu=0 (random games, every round); momentum on hand cases
import torch as th, yaml
from aimanager.manager.rule import run_rule
R = '/opt/algorithmic-institutions/configs/managers/rule_based/'
v7 = compile(yaml.safe_load(open(R+'pf-20261106-11-v7.yml'))['code'], 'v7', 'exec')
v3 = compile(yaml.safe_load(open(R+'pf-20261106-11-v3.yml'))['code'], 'v3', 'exec')
for t in range(24):
    th.manual_seed(t)
    x = dict(c=th.randint(0, 21, (64, 8, t+1)).float(), valid=(th.rand(64, 8, t+1) > 0.1).float(), p=th.randint(0, 31, (64, 8, t)).float(), in_group=(th.rand(64, 8, t+1) > 0.5).float(), t=th.tensor(float(t)))
    pr = dict(rate=7.7, delta=-1.4, decay=0.5, mid=1.0)
    assert th.allclose(run_rule(v7, dict(pr, mu=0.0, theta=2.0), x), run_rule(v3, pr, x))
pr = dict(rate=1.0, delta=0.0, decay=1.0, mid=1.0, mu=1.0, theta=2.0)
g = th.zeros(1, 8, 2); g[:, :4] = 1
c = th.tensor([[12., 6.], [6., 6.], [10., 14.], [10., 14.]]).reshape(1, 4, 2); c = th.cat([c, th.zeros(1, 4, 2)], 1)
x = dict(c=c, valid=th.ones(1, 8, 2), p=th.zeros(1, 8, 1), in_group=g, t=th.tensor(1.))
print('dropper 12->6 vs steady 6 (norm ~11.3/8.7):', run_rule(v7, pr, x)[0, :4].tolist())
x0 = dict(c=c[..., :1], valid=th.ones(1, 8, 1), p=th.zeros(1, 8, 0), in_group=g[..., :1], t=th.tensor(0.))
print('round 0:', run_rule(v7, pr, x0)[0, :4].tolist())
print('v7 == v3 at mu 0: ok')
