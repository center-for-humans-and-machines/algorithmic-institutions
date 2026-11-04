"""v5: gain c[t+3]-c[t] from punishment by the player's cumulative earlier punishment (all rounds
< t, either manager) and by recency; v4 sweep (or the version in argv), my group's stayers."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-7')
from v5_panel import load
pd.set_option('display.width', 250)
r = load(sys.argv[1] if len(sys.argv) > 1 else 'pf-20261104-7-v4')
c, p, v, g = r['c'].astype(float), r['p'].astype(float), r['v'], r['g']
H = 3
rows = []
cum = np.zeros_like(p[:, :, 0])
for t in range(0, 20):
    ok = v[:, :, t] & v[:, :, t + H]
    for s in range(t, t + H + 1):
        ok &= g[:, :, s]
    e, i = np.nonzero(ok)
    last = p[e, i, t - 1] if t > 0 else np.zeros(len(e))
    rows.append(pd.DataFrame({'t': t, 'c': c[e, i, t], 'p': p[e, i, t], 'cum': cum[e, i], 'last': last,
                              'cH': c[e, i, t + H]}))
    cum = cum + p[:, :, t]
d = pd.concat(rows)
d = d[d.c <= 11]
d['gH'] = d.cH - d.c
d['pb'] = pd.cut(d.p, [-1, 0, 5, 15, 30], labels=['0', '1-5', '6-15', '16-30'])
d['cumb'] = pd.cut(d.cum, [-1, 0, 15, 40, 80, 1000], labels=['0', '1-15', '16-40', '41-80', '81+'])
d['cb'] = pd.cut(d.c, [-1, 2, 5, 8, 11], labels=['0-2', '3-5', '6-8', '9-11'])
d['rec'] = np.where(d.cum == 0, 'never', np.where(d['last'] > 0, 'last_round', 'earlier_only'))
for key in ['cumb', 'rec']:
    T = d.groupby(['cb', key, 'pb'], observed=True).gH.agg(['mean', 'size'])
    m = T['mean'].unstack('pb'); n = T['size'].unstack('pb')
    m['eff_6_15'] = m['6-15'] - m['0']; m['eff_16_30'] = m['16-30'] - m['0']
    print('==== gain to t+3 by', key); print(m.round(2).to_string()); print(n.to_string())
# how much of the spending goes to players already punished before (my group's rule only)
print('share of my-group punishment points dealt to players with cum>0:',
      (d.p * (d.cum > 0)).sum() / d.p.sum())
