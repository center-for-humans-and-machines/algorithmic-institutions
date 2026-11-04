"""Run v3's code on a toy game (group 0: players 0-3 contributing 0, 5, 10, 20; players
4-7 in the other group): punishment per round of players 0-3, and of a lone member."""
import torch as th, yaml
cfg = yaml.safe_load(open('configs/managers/rule_based/pf-20261104-12-v3.yml'))
def run(params, grp, cv):
    rows = []
    for t in range(24):
        cvals = th.tensor(cv).view(1, 8, 1).repeat(1, 1, t + 1)
        ig = th.tensor(grp).view(1, 8, 1).repeat(1, 1, t + 1)
        env = dict(c=cvals, valid=th.ones_like(cvals), in_group=ig, p=th.zeros(1, 8, t),
                   t=th.tensor(float(t)), th=th, **{k: th.tensor(float(v)) for k, v in params.items()})
        exec(cfg['code'], {}, env)
        rows.append(env['punishment'][0, :4].clamp(0, 30).floor().int().tolist())
    print(params); print(' '.join(f'{t}:{r}' for t, r in enumerate(rows) if t % 4 < 2))
four = [1., 1, 1, 1, 0, 0, 0, 0]; cv = [0., 5, 10, 20, 3, 3, 3, 3]
run(dict(thr=10., peer=0., slope=5.8, decay=0.6, follow=1.), four, cv)   # v2's best
run(dict(thr=10., peer=1., slope=1.5, decay=1., follow=0.5), four, cv)
run(dict(thr=10., peer=1., slope=2., decay=1., follow=0.), [1., 0, 0, 0, 0, 0, 0, 0], cv)
