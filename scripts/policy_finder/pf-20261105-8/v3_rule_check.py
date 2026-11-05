"""v3: evaluate the v3 rule's code on a toy round to check the phases."""
import torch as th, yaml
cfg = yaml.safe_load(open('configs/managers/rule_based/pf-20261105-8-v3.yml'))
prm = dict(standard=19.3, dose=33.4, curvature=2.23, horizon=9, floor=5, tail=5)
for t in (0, 8, 9, 13, 14):
    c = th.tensor([[[0.], [4.], [10.], [15.]]]).repeat(1, 1, t + 1); valid = th.ones_like(c); p = th.zeros(1, 4, t)
    env = dict(th=th, c=c, valid=valid, p=p, t=th.tensor(float(t)), **{k: th.tensor(float(v)) for k, v in prm.items()})
    exec(cfg['code'], env); print(t, env['punishment'].floor().tolist())
