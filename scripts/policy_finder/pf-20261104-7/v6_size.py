"""v6: does my group's size change what punishment costs in members at a switch decision?
v5 sweep panel; my group = agent_group 0 (g True). For each decision round d (3,7,11,15,19):
size before (members in round d), size after (round d+1), mean punishment per member in my group
over rounds d-3..d, per-capita cg gap in round d."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-7')
from v5_panel import load
pd.set_option('display.width', 250)
r = load(sys.argv[1] if len(sys.argv) > 1 else 'pf-20261104-7-v5')
c, p, v, g = r['c'].astype(float), r['p'].astype(float), r['v'].astype(float), r['g'].astype(float)
o = 1 - g
def cg(mask):
    n = (mask * v).sum(1)
    return np.where(n > 0, (1.6 * (c * mask * v).sum(1) - (p * mask * v).sum(1)) / np.maximum(n, 1), 0)
cg_m, cg_o = cg(g), cg(o)
size = g.sum(1)
rows = []
for d in [3, 7, 11, 15, 19]:
    pm = (p[:, :, d-3:d+1] * g[:, :, d-3:d+1]).sum((1, 2)) / np.maximum(g[:, :, d-3:d+1].sum((1, 2)), 1)
    rows.append(pd.DataFrame({'d': d, 'n0': size[:, d], 'n1': size[:, d+1], 'pun': pm,
                              'gap': cg_m[:, d] - cg_o[:, d], 'gap3': (cg_m[:, d-3:d+1] - cg_o[:, d-3:d+1]).mean(1)}))
x = pd.concat(rows)
x['dn'] = x.n1 - x.n0
x['empty'] = (x.n1 == 0) & (x.n0 > 0)
x['pb'] = pd.cut(x.pun, [-1, 0.5, 3, 6, 10, 31])
x['gb'] = pd.cut(x.gap3, [-99, -10, -4, 0, 4, 10, 99])
for d in [3, 7]:
    y = x[x.d == d]
    print(f'=== decision round {d}')
    print('size change by size x punishment\n', y.pivot_table(index='n0', columns='pb', values='dn', aggfunc='mean', observed=True).round(2))
    print('P(empty after) by size x punishment\n', y.pivot_table(index='n0', columns='pb', values='empty', aggfunc='mean', observed=True).round(3))
    print('P(empty after) by size x gap3\n', y.pivot_table(index='n0', columns='gb', values='empty', aggfunc='mean', observed=True).round(3))
    print('size change by size x gap3\n', y.pivot_table(index='n0', columns='gb', values='dn', aggfunc='mean', observed=True).round(2))
    print('counts by size\n', y.n0.value_counts().sort_index().to_dict())
