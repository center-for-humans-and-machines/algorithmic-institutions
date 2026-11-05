"""v6: the yml rule on hand-made inputs (dose table per contribution, timeouts, phases),
v5 nested as a special case, and the design's spread of dose tables."""
import sys, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
sys.path.insert(0, '/opt/algorithmic-institutions/src')
from v4_sim import rule_fn, ROOT
from aimanager.manager.rule import read_rule, sobol_design
Y = 'configs/managers/rule_based/pf-20261105-3-v6.yml'
cs = th.arange(20.).reshape(5, 4)


def table(prm, t=0):
    f = rule_fn(Y, prm)
    c = cs.unsqueeze(-1).repeat(1, 1, t + 1); v = th.ones_like(c)
    return f(c, v, th.zeros(5, 4, t), t).clamp(0, 30).floor().reshape(-1).int().tolist()


v5 = dict(pmax=32.0, d10=7.0, d16=7.0 * 0.4 ** np.log2(32 / 7), horizon=7, lthr=4, lstop=15)
print('v5-equivalent (d16 %.2f):' % v5['d16'], table(v5))
f5 = rule_fn('configs/managers/rule_based/pf-20261105-3-v5.yml', dict(pmax=32.0, d10=7.0, horizon=7, lthr=4, lstop=15))
c = cs.unsqueeze(-1); print('v5 rule itself         :', f5(c, th.ones_like(c), th.zeros(5, 4, 0), 0).clamp(0, 30).floor().reshape(-1).int().tolist())
for d16 in (0.3, 1.0, 2.0, 3.5):
    print('d10 7 d16', d16, table(dict(v5, d16=d16)))
print('d10 4.5 d16 3.5', table(dict(v5, d10=4.5, d16=3.5)))
for t in (6, 7, 14, 15):
    print('t', t, table(v5, t))
# timeout
f = rule_fn(Y, v5)
c = th.tensor([[0., 5, 15, 20]]).unsqueeze(-1); v = th.ones_like(c); v[0, 0] = 0
print('timeout of player 0:', f(c * v, v, th.zeros(1, 4, 0), 0).floor().tolist())
rule, _ = read_rule(ROOT + Y, 'exp1')
pts = sobol_design(rule, 256)
d = np.array([[p['d10'], p['d16'], p['horizon'], p['lthr'], p['lstop']] for p in pts])
print('design', len(pts), 'corr d10-d16', np.corrcoef(d[:, 0], d[:, 1])[0, 1].round(3),
      'horizon counts', np.bincount(d[:, 2].astype(int))[4:], 'lthr', np.bincount(d[:, 3].astype(int))[2:], 'lstop', np.bincount(d[:, 4].astype(int))[12:])
