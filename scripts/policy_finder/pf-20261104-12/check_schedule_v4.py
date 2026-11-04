"""Run v4's code on toy games: punishment per punished round of the group's members
(contributions 0, 5, 10, 20, ...), for groups of 4, 2 and 6, with size 0 (= v3) and size 1."""
import torch as th, yaml
cfg = yaml.safe_load(open('configs/managers/rule_based/pf-20261104-12-v4.yml'))
def run(params, grp, cv):
    rows = []
    for t in range(24):
        cvals = th.tensor(cv).view(1, 8, 1).repeat(1, 1, t + 1)
        ig = th.tensor(grp).view(1, 8, 1).repeat(1, 1, t + 1)
        env = dict(c=cvals, valid=th.ones_like(cvals), in_group=ig, p=th.zeros(1, 8, t),
                   t=th.tensor(float(t)), th=th, **{k: th.tensor(float(v)) for k, v in params.items()})
        exec(cfg['code'], {}, env)
        pun = env['punishment'][0].clamp(0, 30).floor().int() * th.tensor(grp).int()
        rows.append([x for x, gg in zip(pun.tolist(), grp) if gg])
    print(params, 'group', grp); print(' '.join(f'{t}:{r}' for t, r in enumerate(rows) if t % 4 < 2 and t < 20))
v3 = dict(thr=9.8, peer=0.47, slope=11.95, decay=0.46, follow=0.78)
cv = [0., 5, 10, 20, 3, 15, 12, 8]
for s in (0., 1.):
    for grp in ([1., 1, 1, 1, 0, 0, 0, 0], [1., 1, 0, 0, 0, 0, 0, 0], [1., 1, 1, 1, 1, 1, 0, 0]):
        run(dict(v3, size=s), grp, cv)
