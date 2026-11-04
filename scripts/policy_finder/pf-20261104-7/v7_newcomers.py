"""v7: newcomers after a switch. For each decision d (3,7,11,15,19) and my group (agent_group 0):
players in my group at d+1 but not at d (newcomers) vs players in it at both (incumbents).
Their contributions at d (before arrival, in the other group), d+1, d+2, d+4; incumbents'
change from d to d+2 by the newcomers' contribution gap; and punishment received at d+1."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-7')
from v5_panel import load
pd.set_option('display.width', 250)
r = load(sys.argv[1] if len(sys.argv) > 1 else 'pf-20261104-7-v6')
c, p, v, g = r['c'].astype(float), r['p'].astype(float), r['v'].astype(float), r['g'].astype(float)
rows = []
for d in [3, 7, 11, 15, 19]:
    new = (g[:, :, d+1] == 1) & (g[:, :, d] == 0)
    inc = (g[:, :, d+1] == 1) & (g[:, :, d] == 1)
    for nm, m in [('new', new), ('inc', inc)]:
        e, pl = np.nonzero(m)
        k = min(d + 4, 23)
        rows.append(pd.DataFrame({'d': d, 'who': nm, 'c_d': c[e, pl, d], 'c1': c[e, pl, d+1], 'c2': c[e, pl, d+2],
                                  'c4': c[e, pl, k], 'p1': p[e, pl, d+1], 'p_d': p[e, pl, d], 'n1': g[e, :, d+1].sum(1),
                                  'ep': e, 'pl': pl, 'stay4': g[e, pl, k]}))
x = pd.concat(rows)
print('counts per episode', x.groupby(['d', 'who']).size().unstack() / c.shape[0])
print(x.groupby(['d', 'who'])[['c_d', 'c1', 'c2', 'c4', 'p_d', 'p1']].mean().round(2))
# incumbents' change by newcomers' share and contribution gap
rows = []
for d in [7, 11, 15, 19]:
    new = (g[:, :, d+1] == 1) & (g[:, :, d] == 0)
    inc = (g[:, :, d+1] == 1) & (g[:, :, d] == 1)
    ni, nn = inc.sum(1), new.sum(1)
    ok = (ni > 0)
    ci_d = (c[:, :, d] * inc).sum(1) / np.maximum(ni, 1)
    ci_3 = (c[:, :, d+3] * inc).sum(1) / np.maximum(ni, 1)
    cn_1 = (c[:, :, d+1] * new).sum(1) / np.maximum(nn, 1)
    pin = (p[:, :, d+1:d+4] * g[:, :, d+1:d+4]).sum((1, 2))
    rows.append(pd.DataFrame({'d': d, 'ni': ni, 'nn': nn, 'ci_d': ci_d, 'dci': ci_3 - ci_d,
                              'gap': np.where(nn > 0, cn_1 - ci_d, np.nan), 'pun': pin})[ok])
y = pd.concat(rows)
y['has_new'] = y.nn > 0
y['gb'] = pd.cut(y.gap, [-21, -8, -4, -1, 1, 4, 21])
print('incumbents change d -> d+3 (unpunished windows only, pun==0)')
z = y[y.pun == 0]
print(z.groupby(['d', 'has_new']).dci.agg(['mean', 'count']).round(2))
print(z.groupby('gb', observed=True).dci.agg(['mean', 'count']).round(2))
print(z.groupby(['d'], observed=True).ci_d.mean().round(2))
