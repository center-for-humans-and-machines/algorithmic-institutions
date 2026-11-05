# v4: v4 == v3 at the same params; decay=0 punishes block 0 only (rounds 0-1); hand case
import torch as th, yaml
from aimanager.manager.rule import run_rule
R = '/opt/algorithmic-institutions/configs/managers/rule_based/'
v4 = compile(yaml.safe_load(open(R+'pf-20261106-11-v4.yml'))['code'], 'v4', 'exec')
v3 = compile(yaml.safe_load(open(R+'pf-20261106-11-v3.yml'))['code'], 'v3', 'exec')
g = th.tensor([[1., 1, 1, 1, 0, 0, 0, 0]]).unsqueeze(-1)
for t in range(24):
    th.manual_seed(t)
    x = dict(c=th.randint(0, 21, (64, 8, t+1)).float(), valid=th.ones(64, 8, t+1), p=th.zeros(64, 8, t), in_group=g.repeat(64, 1, t+1), t=th.tensor(float(t)))
    pr = dict(rate=6.6, delta=-1.3, decay=0.4, mid=1.2)
    assert th.allclose(run_rule(v4, pr, x), run_rule(v3, pr, x))
    x['c'][..., -1] = th.tensor([0., 10, 10, 20, 0, 0, 0, 0])
    print(t, run_rule(v4, dict(rate=3.0, delta=-1.0, decay=0.0, mid=1.5), x)[0, :4].tolist())
