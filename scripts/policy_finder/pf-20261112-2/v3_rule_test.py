"""v3: run the v3 rule on a hand-made history in a fresh process: who is fined when."""
import torch as th
from aimanager.manager.rule import read_rule, run_rule
rule = read_rule('configs/managers/rule_based/pf-20261112-2-v3.yml')
rule, code = rule
params = dict(thr=8.0, lo=1, fine=3.0, w=2, first=1)
contrib = th.tensor([0., 1., 5., 7., 8., 20., 3., 2.])
for t in [0, 3, 4, 5, 6, 7, 16, 18, 19, 20]:
    T = t + 1
    c = contrib[None, :, None].repeat(1, 1, T)
    valid = th.ones_like(c); valid[0, 7, -1] = 0
    ing = th.ones_like(c); ing[0, 6] = 0
    p = th.zeros(1, 8, t)
    out = run_rule(code, {k: float(v) for k, v in params.items()}, dict(c=c, valid=valid, p=p, in_group=ing, t=th.tensor(float(t))))
    print(t, out.tolist() if hasattr(out, 'tolist') else out)
