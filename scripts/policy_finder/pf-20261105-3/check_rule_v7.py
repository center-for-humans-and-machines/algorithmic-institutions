"""v7: the yml rule on hand-made inputs (dose table, phases, timeouts) and the design."""
import sys, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
sys.path.insert(0, '/opt/algorithmic-institutions/src')
from v4_sim import rule_fn, ROOT
from aimanager.manager.rule import read_rule, sobol_design
Y = 'configs/managers/rule_based/pf-20261105-3-v7.yml'
cs = th.arange(20.).reshape(5, 4)


def table(prm, t=0):
    f = rule_fn(Y, prm)
    c = cs.unsqueeze(-1).repeat(1, 1, t + 1); v = th.ones_like(c)
    return f(c, v, th.zeros(5, 4, t), t).clamp(0, 30).floor().reshape(-1).int().tolist()


pk = dict(pmax=32.0, d10=5.7, d16=2.6, horizon=6, lthr=4, lstop=14)
for t in (0, 5, 6, 13, 14, 23):
    print('t', t, table(pk, t))
for d10, d16 in ((4.5, 1.0), (4.5, 4.0), (7.5, 1.0), (7.5, 4.0)):
    print('d10', d10, 'd16', d16, table(dict(pk, d10=d10, d16=d16)))
f = rule_fn(Y, pk)
c = th.tensor([[0., 5, 15, 20]]).unsqueeze(-1); v = th.ones_like(c); v[0, 0] = 0
print('timeout of player 0:', f(c * v, v, th.zeros(1, 4, 0), 0).floor().tolist())
rule, _ = read_rule(ROOT + Y, 'exp1')
pts = sobol_design(rule, 256)
d = np.array([[p['d10'], p['d16'], p['horizon'], p['lthr']] for p in pts])
print('design', len(pts), 'corr d10-d16', np.corrcoef(d[:, 0], d[:, 1])[0, 1].round(3),
      'horizon counts', np.bincount(d[:, 2].astype(int))[5:], 'lthr', np.bincount(d[:, 3].astype(int))[2:])
