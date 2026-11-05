"""v6: run the v6 rule at the fitted peak on a toy history and print the punishments."""
import torch as th, yaml
cfg = yaml.safe_load(open('configs/managers/rule_based/pf-20261105-8-v6.yml'))
prm = dict(free=28.0, mid=6.6, horizon=7, end=14, floor=2.0)
cs = th.tensor([[0., 1, 2, 4, 10, 15, 18, 20]]).reshape(2, 4)[None]  # 1 game, two rows -> use 8 players as 2 games
cs = th.tensor([0., 1, 2, 4, 10, 15, 18, 20]).reshape(2, 4)
for t in (0, 6, 7, 13, 14):
    env = dict(th=th, c=cs[:, :, None].repeat(1, 1, t + 1), valid=th.ones(2, 4, t + 1), p=th.zeros(2, 4, t), t=th.tensor(float(t)))
    env.update({k: th.tensor(float(v)) for k, v in prm.items()})
    exec(cfg['code'], env)
    print(t, th.floor(th.clamp(env['punishment'], 0, 30)).reshape(-1).tolist())
