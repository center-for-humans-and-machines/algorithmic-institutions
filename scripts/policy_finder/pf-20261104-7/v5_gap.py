"""v5: punishment effect on c[t+3]-c[t] by rounds since the player's last punishment by my rule
(1, 2, 3, 4+, never), v4 sweep, my group's stayers with c_t <= 8."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-7')
from v5_panel import load
pd.set_option('display.width', 250)
r = load(sys.argv[1] if len(sys.argv) > 1 else 'pf-20261104-7-v4')
c, p, v, g = r['c'].astype(float), r['p'].astype(float), r['v'], r['g']
H = 3
rows = []
lastp = np.full(p[:, :, 0].shape, -99)
for t in range(0, 20):
    ok = v[:, :, t] & v[:, :, t + H]
    for s in range(t, t + H + 1):
        ok &= g[:, :, s]
    e, i = np.nonzero(ok)
    rows.append(pd.DataFrame({'c': c[e, i, t], 'p': p[e, i, t], 'since': t - lastp[e, i], 'cH': c[e, i, t + H]}))
    lastp = np.where((p[:, :, t] > 0) & g[:, :, t], t, lastp)
d = pd.concat(rows)
d = d[d.c <= 8]
d['gH'] = d.cH - d.c
d['sb'] = pd.cut(d.since, [0, 1, 2, 3, 30, 1000], labels=['1', '2', '3', '4+', 'never'])
d['pb'] = pd.cut(d.p, [-1, 0, 5, 15, 30], labels=['0', '1-5', '6-15', '16-30'])
d['cb'] = pd.cut(d.c, [-1, 2, 5, 8], labels=['0-2', '3-5', '6-8'])
T = d.groupby(['cb', 'sb', 'pb'], observed=True).gH.mean().unstack('pb')
T['eff_6_15'] = T['6-15'] - T['0']; T['eff_16_30'] = T['16-30'] - T['0']
print(T.round(2).to_string())
