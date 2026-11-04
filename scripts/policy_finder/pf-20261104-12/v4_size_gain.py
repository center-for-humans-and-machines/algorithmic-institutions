"""v4: at block-first rounds 4, 8, 12, for members of the rule's group contributing <= 10:
slope of their later contribution gain (rounds t+1..t+7 mean minus c_t, members staying
throughout) and of leaving at the next decision on the punishment received, by group size."""
import sys, numpy as np, pandas as pd
rows = []
for f in sys.argv[1:]:
    z = np.load(f); c, p, g, v = (z[k].astype(np.int16) for k in 'cpgv')
    for t in (4, 8, 12):
        n = (g[..., t] == 0).sum(-1, keepdims=True).repeat(8, -1)
        m = (g[..., t] == 0) & (v[..., t] == 1) & (c[..., t] <= 10)
        sa = (g[..., t:t+8] == 0).all(-1)
        rows.append(pd.DataFrame({'n': n[m], 'c': c[..., t][m], 'p': p[..., t][m], 'gain': (c[..., t+1:t+8].mean(-1) - c[..., t])[m],
                                  'sa': sa[m], 'left': (g[..., t+4] != 0)[m]}))
d = pd.concat(rows)
def sl(x, y):
    x = x - x.mean(); return (x*(y - y.mean())).sum()/max((x*x).sum(), 1e-9)
d['cb'] = pd.cut(d.c, [-1, 4, 10])
for (cb, n), x in d.groupby(['cb', 'n'], observed=True):
    s = x[x.sa]
    print(cb, n, len(x), 'gain/pt %.3f' % sl(s.p.astype(float), s.gain), 'leave pp/pt %.3f' % (100*sl(x.p.astype(float), x.left.astype(float))))
