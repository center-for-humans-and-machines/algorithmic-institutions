"""v3: a mock sweep on the surrogates over the planned v3 ranges (T, A, B, S, N), to see where the
optimum lies and whether it is interior. Uses the real sobol_design of the rule config."""
import sys, numpy as np, torch as th; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from screen import *
from aimanager.manager.rule import read_rule, sobol_design
import inspect

rule = read_rule('configs/managers/rule_based/pf-20261105-11-v3.yml', setting='exp1')[0]
print(inspect.signature(sobol_design))
pts = sobol_design(rule, 256)


def mk(T, A, B, S, N):
    def f(c, v, p, t):
        if t > S: return zeros(c)
        s = th.relu(T - c[..., -1])
        x = th.where(s > 0, A + B * s, zeros(c))
        if t == 0: return x
        return th.where((p > 0).float().sum(-1) >= N, zeros(c), x)
    return f


rows = []
for i, pt in enumerate(pts):
    r = ev(mk(pt['T'], pt['A'], pt['B'], pt['S'], pt['N']), B=1000, seed=i)
    rows.append([pt[k] for k in 'TABSN'] + [(r['mlp']['score'] + r['gru']['score']) / 2])
rows = np.array(rows)
np.save('/tmp/pf3/mock_sweep.npy', rows)
o = rows[np.argsort(-rows[:, -1])]
print('top 15 (T A B S N score)'); print(np.round(o[:15], 2))
for j, k in enumerate('TABSN'):
    top = o[:25, j]
    print(k, 'top-25 range', round(top.min(), 2), round(top.max(), 2), 'median', round(np.median(top), 2))
