"""v5: punishments of the v5 rule at the pooled peak, by round and contribution."""
import torch as th, yaml
r = yaml.safe_load(open('configs/managers/rule_based/pf-20261105-8-v5.yml'))
prm = dict(standard=22.7, rate=0.112, dose=28.7, horizon=7, end=14, floor=2.0)
cs = th.tensor([[0., 1, 2, 4, 10, 15, 18, 20]])
for t in [0, 6, 7, 13, 14, 23]:
    env = dict(th=th, c=cs[..., None].repeat(1, 1, t + 1), valid=th.ones(1, 8, t + 1), p=th.zeros(1, 8, t), t=th.tensor(t))
    env.update({k: th.tensor(v) for k, v in prm.items()}); exec(r['code'], env)
    print(t, th.floor(th.clamp(env['punishment'], 0, 30)).tolist())
