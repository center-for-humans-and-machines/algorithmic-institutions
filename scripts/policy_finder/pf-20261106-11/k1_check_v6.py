# v6: v6 == v4 at sigma=0 (any group composition); size factor on hand cases (group of 2, 4, 8)
import torch as th, yaml
from aimanager.manager.rule import run_rule
R = '/opt/algorithmic-institutions/configs/managers/rule_based/'
v6 = compile(yaml.safe_load(open(R+'pf-20261106-11-v6.yml'))['code'], 'v6', 'exec')
v4 = compile(yaml.safe_load(open(R+'pf-20261106-11-v4.yml'))['code'], 'v4', 'exec')
for t in range(24):
    th.manual_seed(t)
    x = dict(c=th.randint(0, 21, (64, 8, t+1)).float(), valid=(th.rand(64, 8, t+1) > 0.1).float(), p=th.randint(0, 31, (64, 8, t)).float(), in_group=(th.rand(64, 8, t+1) > 0.5).float(), t=th.tensor(float(t)))
    pr = dict(rate=7.7, delta=-1.4, decay=0.5, mid=1.0)
    assert th.allclose(run_rule(v6, dict(pr, sigma=0.0), x), run_rule(v4, pr, x))
pr = dict(rate=1.0, delta=0.0, decay=1.0, mid=1.0, sigma=1.0)
for n in [2, 4, 8]:
    g = th.zeros(1, 8, 5); g[:, :n] = 1
    c = th.full((1, 8, 5), 10.); c[0, 0, -1] = 4
    x = dict(c=c, valid=th.ones(1, 8, 5), p=th.zeros(1, 8, 4), in_group=g, t=th.tensor(4.))
    print(n, run_rule(v6, pr, x)[0, :n].tolist())
print('v6 == v4 at sigma 0: ok')
