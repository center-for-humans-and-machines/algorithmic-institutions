"""v5: the yml rule on hand-made inputs, its surrogate score at the expected best vs v4's
peak, and the whole Sobol design played on the refit surrogate (a preview of the sweep)."""
import sys, json, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
sys.path.insert(0, '/opt/algorithmic-institutions/src')
import v5_sim
from v4_sim import simulate, rule_fn, ROOT
from aimanager.manager.rule import read_rule, sobol_design
th.set_num_threads(2)
Y = 'configs/managers/rule_based/pf-20261105-3-v5.yml'
best = dict(pmax=30.0, d10=7.0, horizon=7, lthr=4, lstop=15)
f = rule_fn(Y, best)
c = th.tensor([[0., 2, 4, 8], [10, 12, 16, 20]]).unsqueeze(-1).repeat(1, 1, 24)
v = th.ones_like(c); v[0, 1] = 0
for t in (0, 6, 7, 14, 15, 23):
    print(t, f(c[..., :t + 1], v[..., :t + 1], th.zeros(2, 4, t), t).floor().tolist())
S = (901, 902, 903)
print('best', np.mean([simulate(f, n=3000, seed=s)[0] for s in S]))
v4 = rule_fn('configs/managers/rule_based/pf-20261105-3-v4.yml', dict(thr=24, pmax=30.08, gamma=2.93, horizon=7, lthr=4, lstop=15))
print('v4 peak', np.mean([simulate(v4, n=3000, seed=s)[0] for s in S]))
rule, _ = read_rule(ROOT + Y, 'exp1')
pts = sobol_design(rule, 256)
res = []
for i, prm in enumerate(pts):
    s, se = simulate(rule_fn(Y, prm), n=500, seed=1000 + i)
    res.append((s, prm))
res.sort(key=lambda x: -x[0])
sc = np.array([r[0] for r in res])
print('design: max', sc.max(), 'q90', np.quantile(sc, .9), 'median', np.median(sc), 'min', sc.min())
for s, prm in res[:12]:
    print(round(s, 1), {k: round(x, 2) if isinstance(x, float) else x for k, x in prm.items()})
