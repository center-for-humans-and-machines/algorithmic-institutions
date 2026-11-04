"""v3: on cached sweep arrays (v1+v2), for members of the rule's group at the punished
rounds (block position 0 or 1): slope of later contribution, and of leaving at the next
decision, on the punishment received, within cells of own contribution, change from the
previous round, newcomer status and block. Slopes by OLS of y on p within each cell."""
import sys, numpy as np, pandas as pd
rows = []
for f in sys.argv[1:]:
    z = np.load(f); c, p, g, v = (z[k].astype(np.int16) for k in 'cpgv')
    for t in range(20):
        pos = t % 4
        if pos > 1: continue
        dec = t - pos + 3
        inn = (g[..., t] == 0) & (v[..., t] == 1)
        new = np.zeros_like(inn) if t < 4 else (g[..., t-pos-1] != 0)   # arrived at this block
        prev = c[..., t-1] if t > 0 else c[..., t]
        stay = (g[..., dec+1] == 0) if dec < 23 else np.ones_like(inn)
        # mean contribution over rounds t+1..dec+4 (the rest of this block and the next) when staying
        hor = list(range(t+1, min(dec+5, 24)))
        fut = c[..., hor].mean(-1)
        stay_all = (g[..., hor] == 0).all(-1)
        m = inn
        rows.append(pd.DataFrame({'t': t, 'b': t//4, 'pos': pos, 'c': c[..., t][m], 'dc': (c[..., t]-prev)[m],
            'new': new[m], 'p': p[..., t][m], 'fut': fut[m], 'sa': stay_all[m], 'left': ~stay[m],
            'oth': ((c[..., t]*(g[..., t]==0)).sum(-1, keepdims=True).repeat(8, -1)[m] - c[..., t][m]) / np.maximum((g[..., t]==0).sum(-1, keepdims=True).repeat(8,-1)[m]-1, 1)}))
d = pd.concat(rows)
d['gain'] = d.fut - d.c
d['cb'] = pd.cut(d.c, [-1, 0, 3, 6, 9, 12, 20])
d['db'] = pd.cut(d.dc, [-21, -4, -1, 0, 3, 21], labels=['fell>3', 'fell1-3', 'same', 'rose1-3', 'rose>3'])
def slope(x, y):
    x = x - x.mean(); return (x*(y - y.mean())).sum() / max((x*x).sum(), 1e-9)
def tab(by, d):
    out = []
    for k, x in d.groupby(by, observed=True):
        if len(x) < 2000: continue
        s = x[x.sa]
        out.append(dict(zip(by, k if isinstance(k, tuple) else (k,)), n=len(x), pmean=x.p.mean(),
            gain_per_pt=slope(s.p, s.gain), leave_per_pt=slope(x.p.astype(float), x.left.astype(float))*100,
            leave0=x[x.p == 0].left.mean()*100))
    return pd.DataFrame(out).round(3)
pd.set_option('display.width', 200)
print('by own c and change'); print(tab(['cb', 'db'], d).to_string())
print('by own c and newcomer (blocks 1-4)'); print(tab(['cb', 'new'], d[d.b > 0]).to_string())
print('by block and pos, c<=9'); print(tab(['b', 'pos'], d[d.c <= 9]).to_string())
d['ob'] = pd.cut(d.oth, [-1, 6, 9, 12, 15, 20])
print('by own c and others mean, c<=9'); print(tab(['cb', 'ob'], d[d.c <= 9]).to_string())
