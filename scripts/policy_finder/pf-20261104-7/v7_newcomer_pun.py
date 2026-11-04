"""v7: newcomers' response to punishment in their first round with me. Newcomers at decisions
d = 7, 11, 15 (arrive at round d+1); by their first-round contribution c1 and punishment p1:
contribution at d+4 (before the next switch), still in my group after the next decision (d+5),
and the incumbents' mean change d -> d+4."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-7')
from v5_panel import load
pd.set_option('display.width', 250)
r = load(sys.argv[1] if len(sys.argv) > 1 else 'pf-20261104-7-v6')
c, p, v, g = r['c'].astype(float), r['p'].astype(float), r['v'].astype(float), r['g'].astype(float)
rows = []
for d in [7, 11, 15]:
    new = (g[:, :, d+1] == 1) & (g[:, :, d] == 0) & (v[:, :, d+1] == 1)
    inc = (g[:, :, d+1] == 1) & (g[:, :, d] == 1)
    ni = inc.sum(1)
    dci = ((c[:, :, d+4] - c[:, :, d]) * inc).sum(1) / np.maximum(ni, 1)
    e, pl = np.nonzero(new)
    rows.append(pd.DataFrame({'d': d, 'c1': c[e, pl, d+1], 'p1': p[e, pl, d+1], 'pw': p[e, pl, d+1:d+5].sum(1),
                              'c4': c[e, pl, d+4], 'stay': g[e, pl, d+5], 'dci': dci[e], 'ni': ni[e],
                              'n_in': g[e, :, d+1].sum(1), 'n_after': g[e, :, d+5].sum(1)}))
x = pd.concat(rows)
x['cb'] = pd.cut(x.c1, [-1, 3, 7, 10, 20])
x['pb'] = pd.cut(x.p1, [-1, 0, 5, 12, 20, 30])
x['g'] = x.c4 - x.c1
for col in ['g', 'stay', 'dci']:
    print(f'--- {col} by first-round contribution x first-round punishment')
    print(x.pivot_table(index='cb', columns='pb', values=col, aggfunc='mean', observed=True).round(2))
print(x.pivot_table(index='cb', columns='pb', values='g', aggfunc='count', observed=True))
