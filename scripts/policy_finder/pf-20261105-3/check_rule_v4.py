"""v4: the yml rule on hand-made inputs, its surrogate score at the expected best,
and the whole Sobol design played on the surrogate (a preview of the sweep)."""
import sys, json, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
sys.path.insert(0, '/opt/algorithmic-institutions/src')
from v4_sim import simulate, rule_fn, ROOT
from aimanager.manager.rule import read_rule, sobol_design
th.set_num_threads(2)
Y = 'configs/managers/rule_based/pf-20261105-3-v4.yml'
best = dict(thr=21, pmax=31.0, gamma=2.3, horizon=6, lthr=3, lstop=15)
f = rule_fn(Y, best)
c = th.tensor([[0., 2, 4, 8], [12, 16, 19, 20]]).unsqueeze(-1).repeat(1, 1, 24)
v = th.ones_like(c); v[1, 3] = 0
for t in (0, 5, 6, 14, 15, 23):
    print(t, f(c[..., :t + 1], v[..., :t + 1], th.zeros(2, 4, t), t).floor().tolist())
print('best', np.mean([simulate(f, n=3000, seed=s)[0] for s in (401, 402, 403)]))
off = dict(best, lthr=0, horizon=10)
print('late phase off, horizon 10', np.mean([simulate(rule_fn(Y, off), n=3000, seed=s)[0] for s in (401, 402, 403)]))
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
json.dump(res, open('/tmp/v4_design_preview.json', 'w'))
