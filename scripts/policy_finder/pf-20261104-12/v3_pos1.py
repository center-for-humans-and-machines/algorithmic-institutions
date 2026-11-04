"""v3: at block position 1 (rounds 1,5,9,13,17), members of the rule's group who were
punished at position 0: by whether they raised their contribution since position 0,
the slope of their later contribution gain (rounds t+1..t+6) and of their leaving at the
block's decision on the position-1 punishment; and the group-level leave rate of the
*others* as a function of the pos-1 group mean punishment."""
import sys, numpy as np, pandas as pd
rows = []
for f in sys.argv[1:]:
    z = np.load(f); c, p, g, v = (z[k].astype(np.int16) for k in 'cpgv')
    for t in (1, 5, 9, 13, 17):
        dec = t + 2
        inn = (g[..., t] == 0) & (g[..., t-1] == 0) & (v[..., t] == 1)
        hor = list(range(t+1, min(t+7, 24)))
        sa = (g[..., hor] == 0).all(-1)
        m = inn & (p[..., t-1] > 0)
        rows.append(pd.DataFrame({'t': t, 'c0': c[..., t-1][m], 'c1': c[..., t][m], 'p0': p[..., t-1][m], 'p1': p[..., t][m],
            'fut': c[..., hor].mean(-1)[m], 'sa': sa[m], 'left': (g[..., dec+1] != 0)[m]}))
d = pd.concat(rows)
d['rose'] = pd.cut(d.c1 - d.c0, [-21, -1, 0, 3, 21], labels=['fell', 'same', 'rose1-3', 'rose>3'])
d['cb'] = pd.cut(d.c1, [-1, 3, 7, 11, 20])
d['gain'] = d.fut - d.c1
def slope(x, y):
    x = x - x.mean(); return (x*(y - y.mean())).sum() / max((x*x).sum(), 1e-9)
out = []
for k, x in d.groupby(['cb', 'rose'], observed=True):
    if len(x) < 3000: continue
    s = x[x.sa]
    out.append(dict(cb=k[0], rose=k[1], n=len(x), p1=x.p1.mean(), gain_pt=slope(s.p1.astype(float), s.gain),
                    gain_p1zero=s[s.p1 == 0].gain.mean(), leave_pt=100*slope(x.p1.astype(float), x.left.astype(float)), leave0=100*x[x.p1 == 0].left.mean()))
print(pd.DataFrame(out).round(3).to_string())
