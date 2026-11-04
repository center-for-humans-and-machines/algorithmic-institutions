"""v4: dose-response at a punished round t (default 0) for members of the rule's group:
by own contribution bin and punishment bin, the mean contribution over rounds t+1..t+3 and
over rounds 4..11 if still in the group, and the rate of leaving at the next decision."""
import sys, numpy as np, pandas as pd
t = int(sys.argv[1]); files = sys.argv[2:]
rows = []
for f in files:
    z = np.load(f); c, p, g, v = (z[k].astype(np.int16) for k in 'cpgv')
    dec = t - t % 4 + 3
    m = (g[..., t] == 0) & (v[..., t] == 1)
    stay = g[..., dec+1] == 0
    nxt = c[..., t+1:dec+1].mean(-1)
    later = c[..., dec+1:dec+9].mean(-1)
    stay2 = (g[..., dec+1:dec+9] == 0).all(-1)
    oth = ((c[..., t]*(g[..., t] == 0)).sum(-1, keepdims=True) - c[..., t]) / np.maximum((g[..., t] == 0).sum(-1, keepdims=True) - 1, 1)
    rows.append(pd.DataFrame({'c': c[..., t][m], 'p': p[..., t][m], 'nxt': nxt[m], 'later': later[m], 'stay': stay[m], 'stay2': stay2[m], 'oth': oth[m]}))
d = pd.concat(rows)
d['cb'] = pd.cut(d.c, [-1, 2, 5, 8, 11, 20])
d['pb'] = pd.cut(d.p, [-1, 0, 5, 10, 15, 20, 25, 29, 30])
out = d.groupby(['cb', 'pb'], observed=True).apply(lambda x: pd.Series({'n': len(x), 'leave%': 100*(1-x.stay.mean()),
      'nxt': x[x.stay].nxt.mean() if t % 4 < 3 else np.nan, 'later': x[x.stay2].later.mean(), 'oth': x.oth.mean()}))
pd.set_option('display.width', 200)
print(out[out.n >= 300].round(2).to_string())
