"""Run the rule's code on a toy game: which rounds punish, and how much a 0, 5, 10, 20 contributor gets."""
import torch as th, yaml
cfg = yaml.safe_load(open('configs/managers/rule_based/pf-20261104-12.yml'))
for params in [dict(thr=10., slope=0.8, quiet=2, stop=3), dict(thr=10., slope=0.8, quiet=0, stop=1)]:
    rows = []
    for t in range(24):
        cvals = th.tensor([0., 5., 10., 20., 0, 0, 0, 0]).view(1, 8, 1).repeat(1, 1, t + 1)
        env = dict(c=cvals, valid=th.ones_like(cvals), in_group=th.ones_like(cvals), p=th.zeros(1, 8, t),
                   t=th.tensor(float(t)), th=th, **{k: th.tensor(float(v)) for k, v in params.items()})
        exec(cfg['code'], {}, env)
        rows.append(env['punishment'][0, :4].clamp(0, 30).int().tolist())
    print(params); print(' '.join(f'{t}:{r[0]}/{r[1]}' for t, r in enumerate(rows)))
