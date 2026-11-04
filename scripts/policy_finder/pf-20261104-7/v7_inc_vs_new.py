"""v7: same response table for incumbents (in my group at d and d+1) at arrival rounds d+1 =
8, 12, 16, compared with newcomers; plus how many punishment points per episode a newcomer clause
would spend: newcomers whose c is below (others' mean - slack) for slack 0, 2, 4, 6."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-7')
from v5_panel import load
pd.set_option('display.width', 250)
r = load(sys.argv[1] if len(sys.argv) > 1 else 'pf-20261104-7-v6')
c, p, v, g = r['c'].astype(float), r['p'].astype(float), r['v'].astype(float), r['g'].astype(float)
rows = []
for d in [7, 11, 15]:
    for nm, m in [('new', (g[:, :, d+1] == 1) & (g[:, :, d] == 0)), ('inc', (g[:, :, d+1] == 1) & (g[:, :, d] == 1))]:
        m = m & (v[:, :, d+1] == 1)
        e, pl = np.nonzero(m)
        prior = p[e, pl, :d+1].sum(1)
        rows.append(pd.DataFrame({'who': nm, 'c1': c[e, pl, d+1], 'p1': p[e, pl, d+1], 'g': c[e, pl, d+4] - c[e, pl, d+1],
                                  'stay': g[e, pl, d+5], 'prior': prior}))
x = pd.concat(rows)
x['cb'] = pd.cut(x.c1, [-1, 3, 7, 10, 20]); x['pb'] = pd.cut(x.p1, [-1, 0, 5, 12, 20, 30])
for col in ['g', 'stay']:
    print(col); print(x.pivot_table(index=['who', 'cb'], columns='pb', values=col, aggfunc='mean', observed=True).round(2))
# newcomer clause spending and reach, rounds 4..23, newcomers within 1 or 4 rounds of arrival
T = 24
for w in [1, 4]:
    for slack in [0, 2, 4, 6]:
        tot = 0; cnt = 0
        for t in range(4, T):
            lo = max(t - w, 0)
            new = (g[:, :, t] == 1) & ((g[:, :, lo:t] == 0).any(-1)) & (v[:, :, t] == 1)
            mem = (g[:, :, t] == 1) & (v[:, :, t] == 1)
            n = mem.sum(1, keepdims=True)
            others = ((c[:, :, t] * mem).sum(1, keepdims=True) - c[:, :, t] * mem) / np.maximum(n - 1, 1)
            short = np.clip(others - slack - c[:, :, t], 0, None) * new
            tot += np.minimum(short * 3, 30).sum(); cnt += (short > 0).sum()
        print(f'window {w} slack {slack}: newcomer-rounds below bar per episode {cnt/c.shape[0]:.2f}, points at rate 3 per episode {tot/c.shape[0]:.1f}')
