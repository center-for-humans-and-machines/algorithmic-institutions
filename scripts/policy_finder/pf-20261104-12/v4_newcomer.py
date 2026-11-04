"""v4: at a block's first round t (4, 8, 12), members of the rule's group split into
newcomers (in the other group at t-1) and stayers; by own contribution bin and punishment
bin: n, leave rate at the next decision, later contribution (rounds t+4..t+11 if staying)."""
import sys, numpy as np, pandas as pd
rows = []
for f in sys.argv[1:]:
    z = np.load(f); c, p, g, v = (z[k].astype(np.int16) for k in 'cpgv')
    for t in (4, 8, 12):
        m = (g[..., t] == 0) & (v[..., t] == 1)
        new = g[..., t-1] != 0
        stay = g[..., t+4] == 0
        e = min(t+12, 24)
        later = c[..., t+4:e].mean(-1); stay2 = (g[..., t+4:e] == 0).all(-1)
        rows.append(pd.DataFrame({'t': t, 'new': new[m], 'c': c[..., t][m], 'p': p[..., t][m], 'stay': stay[m], 'later': later[m], 'stay2': stay2[m],
                                  'pin': p[..., t-4:t].sum(-1)[m]}))
d = pd.concat(rows)
d['cb'] = pd.cut(d.c, [-1, 4, 8, 11, 20]); d['pb'] = pd.cut(d.p, [-1, 0, 5, 12, 22, 30])
o = d.groupby(['t', 'new', 'cb', 'pb'], observed=True).apply(lambda x: pd.Series({'n': len(x), 'leave%': 100*(1-x.stay.mean()), 'later': x[x.stay2].later.mean(), 'pin': x.pin.mean()}), include_groups=False)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 500)
print(o[o.n >= 400].round(2).unstack('new').to_string())
