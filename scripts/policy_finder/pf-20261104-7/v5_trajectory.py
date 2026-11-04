"""v5: does the contribution gain from a punishment depend on the player's own trajectory?
v4 sweep, my group's stayers. Gain = c[t+h] - c[t] for punished vs unpunished players in the same
(c_t, c_{t-1}) cell; punishment varies across design points (threshold/rate), so within a cell it
is close to random."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-7')
from v5_panel import load
pd.set_option('display.width', 250)
r = load(sys.argv[1] if len(sys.argv) > 1 else 'pf-20261104-7-v4')
c, p, v, g = r['c'].astype(float), r['p'].astype(float), r['v'], r['g']
rows = []
H = 3
for t in range(1, 20):
    ok = g[:, :, t] & g[:, :, t + H] & g[:, :, t - 1] & v[:, :, t] & v[:, :, t - 1] & v[:, :, t + H]
    # stay throughout t..t+H
    for s in range(t, t + H + 1):
        ok &= g[:, :, s]
    e, i = np.nonzero(ok)
    rows.append(pd.DataFrame({'t': t, 'c': c[e, i, t], 'cprev': c[e, i, t - 1], 'pprev': p[e, i, t - 1], 'p': p[e, i, t],
                              'c1': c[e, i, t + 1], 'cH': c[e, i, t + H]}))
d = pd.concat(rows)
d['trend'] = np.sign(d.c - d.cprev).map({-1: 'down', 0: 'flat', 1: 'up'})
d['pb'] = pd.cut(d.p, [-1, 0, 5, 15, 30], labels=['0', '1-5', '6-15', '16-30'])
d['prevp'] = np.where(d.pprev > 0, 'pun_before', 'clean_before')
d['cb'] = pd.cut(d.c, [-1, 2, 5, 8, 11], labels=['0-2', '3-5', '6-8', '9-11'])
d = d[d.c <= 11]
d['g1'] = d.c1 - d.c; d['gH'] = d.cH - d.c
for key in ['trend', 'prevp']:
    print('==== by', key, f'(gain to t+1, gain to t+{H}, N)')
    T = d.groupby(['cb', key, 'pb'], observed=True).agg(g1=('g1', 'mean'), gH=('gH', 'mean'), N=('g1', 'size'))
    print(T.round(2).unstack('pb').to_string())
