"""Run v5's code on toy games: punishment per punished round of the group's members
(contributions 0, 5, 10, 20, ...), for groups of 4, 2 and 6, at v4's final peak and at the corners
of v5's new ranges (slope 50 / decay 0.15 / size 4)."""
import torch as th, yaml
cfg = yaml.safe_load(open('configs/managers/rule_based/pf-20261104-12-v5.yml'))
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
    assert all(sum(r) == 0 for t, r in enumerate(rows) if t % 4 >= 2 or t >= 20)
v4 = dict(thr=8.96, peer=0.54, slope=20, decay=0.31, follow=0.70, size=3.0)
cv = [0., 5, 10, 20, 3, 15, 12, 8]
for params in (v4, dict(v4, slope=50, decay=0.15, size=4.0), dict(v4, slope=5, decay=0.8, size=0.0)):
    for grp in ([1., 1, 1, 1, 0, 0, 0, 0], [1., 1, 0, 0, 0, 0, 0, 0], [1., 1, 1, 1, 1, 1, 0, 0]):
        run(params, grp, cv)
