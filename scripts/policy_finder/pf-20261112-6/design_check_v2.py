"""v2: design coverage of the v2 rule (how many points are zero-equivalent) and a toy run of its code
on hand-made histories, to check eligibility, streak and threshold."""
import numpy as np, pandas as pd, yaml, torch as th
from aimanager.manager.rule import sobol_design, read_rule, run_rule
path = 'configs/managers/rule_based/pf-20261112-6-v2.yml'
rule = yaml.safe_load(open(path))
pts = pd.DataFrame(sobol_design(rule, 256))
print(pts.describe().round(2))
zero = (pts.n_blocks == 0) | (pts.amt < 1)
print('zero-equivalent points:', zero.sum(), 'of', len(pts))
print(pts[~zero].groupby('n_blocks').size().to_dict(), pts.groupby('streak').size().to_dict())
# toy: 1 game, player 0 contributes 0 always, player 1 contributes 0 only now, player 2 contributes 10
def toy(t, params):
    T = t + 1
    c = th.full((1, 8, T), 15.0); c[0, 0] = 0; c[0, 1, -1] = 0; c[0, 2] = 10
    inp = dict(c=c, valid=th.ones_like(c), p=th.zeros(1, 8, t), in_group=th.ones_like(c), t=th.tensor(float(t)))
    loc = dict(th=th, **inp, **{k: th.tensor(float(v)) for k, v in params.items()})
    exec(rule['code'], {}, loc)
    return loc['punishment'][0].tolist()
for t in (0, 3, 4, 8, 16, 20):
    print(t, toy(t, dict(n_blocks=3, thr=2, streak=2, amt=5)), toy(t, dict(n_blocks=5, thr=10, streak=1, amt=5)))
