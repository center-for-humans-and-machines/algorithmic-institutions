"""v7: run the v7 rule on a toy history: players drop from 20 / 12 / stay low / timed out last round."""
import torch as th, yaml
cfg = yaml.safe_load(open('configs/managers/rule_based/pf-20261105-8-v7.yml'))
prm = dict(free=30.0, mid=6.9, horizon=6, high=15.0, end=14)
# game 0: prev 20,20,12,0 -> now 0,10,0,0 ; game 1: prev 20(timed out),18,20,20 -> now 0,18,19,20
prev = th.tensor([[20., 20, 12, 0], [0, 18, 20, 20]]); now = th.tensor([[0., 10, 0, 0], [0, 18, 19, 20]])
pv = th.tensor([[1., 1, 1, 1], [0, 1, 1, 1]])
for t in (0, 5, 6, 10, 13, 14):
    c = th.cat([prev[:, :, None].repeat(1, 1, t), now[:, :, None]], -1)
    v = th.cat([pv[:, :, None].repeat(1, 1, t), th.ones(2, 4, 1)], -1)
    env = dict(th=th, c=c, valid=v, p=th.zeros(2, 4, t), t=th.tensor(float(t)))
    env.update({k: th.tensor(float(x)) for k, x in prm.items()})
    exec(cfg['code'], env)
    print(t, th.floor(th.clamp(env['punishment'], 0, 30)).tolist())
