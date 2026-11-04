"""Run v6's code on toy games: punishment per punished round of the group's members,
in a stingy group (0, 3, 5, 8), a generous one (0, 5, 15, 20) and a pair, at v5's
final peak with herd 0 (must equal v5) and herd 1 / 2.5 / -0.5."""
import torch as th, yaml
cfg = yaml.safe_load(open('configs/managers/rule_based/pf-20261104-12-v6.yml'))
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
    print(params.get('herd'), [x for x, gg in zip(cv, grp) if gg], ' '.join(f'{t}:{r}' for t, r in enumerate(rows) if t % 4 < 2 and t < 20))
    assert all(sum(r) == 0 for t, r in enumerate(rows) if t % 4 >= 2 or t >= 20)
v5 = dict(thr=8.14, peer=0.58, slope=9.73, decay=0.41, follow=1.24)
G4 = [1., 1, 1, 1, 0, 0, 0, 0]
for herd in (0.0, 1.0, 2.5, -0.5):
    for grp, cv in ((G4, [0., 3, 5, 8, 9, 9, 9, 9]), (G4, [0., 5, 15, 20, 9, 9, 9, 9]), ([1., 1, 0, 0, 0, 0, 0, 0], [0., 15, 9, 9, 9, 9, 9, 9])):
        run(dict(v5, herd=herd), grp, cv)
