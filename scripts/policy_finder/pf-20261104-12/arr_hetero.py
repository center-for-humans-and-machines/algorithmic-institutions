"""Heterogeneity of the response: c_{t+1}-c_t of members of the rule's group with c_t<=6,
by punishment bin and (a) round, (b) number of earlier rounds in which the player was
punished, (c) group mean contribution of the others this round. Also: the leave rate at a
decision of members punished only in the block's first two rounds (quiet>=2 points)."""
import sys, numpy as np, pandas as pd
z = np.load(sys.argv[1]); c, p, g, v = (z[k].astype(np.int16) for k in 'cpgv')
quiet = z['params'][:, list(z['keys']).index('quiet')]
npun = np.concatenate([np.zeros_like(p[..., :1]), np.cumsum(p > 0, -1)[..., :-1]], -1)
rows = []
for t in range(23):
    m = (g[..., t] == 0) & (v[..., t] == 1) & (g[..., t+1] == 0) & (c[..., t] <= 6)
    ing = (g[..., t] == 0)
    tot = (c[..., t] * ing).sum(-1, keepdims=True); nn = ing.sum(-1, keepdims=True)
    oth = np.broadcast_to((tot - c[..., t]) / np.maximum(nn - 1, 1), m.shape)
    rows.append(pd.DataFrame({'t': t, 'c': c[..., t][m], 'p': p[..., t][m], 'dc': (c[..., t+1] - c[..., t])[m], 'k': npun[..., t][m], 'oth': oth[m]}))
d = pd.concat(rows)
d['pb'] = pd.cut(d.p, [-1, 0, 10, 20, 30])
d['tb'] = pd.cut(d.t, [-1, 3, 7, 11, 15, 22])
d['kb'] = pd.cut(d.k, [-1, 0, 1, 3, 6, 23])
d['ob'] = pd.cut(d.oth, [-1, 6, 10, 14, 20])
for b in ('tb', 'kb', 'ob'):
    tab = d.groupby([b, 'pb'], observed=True).dc.agg(['mean', 'size'])
    print(tab['mean'].where(tab['size'] > 300).unstack().round(2))
