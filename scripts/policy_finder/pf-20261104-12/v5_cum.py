"""v5: does the answer to a punishment shrink with the punishment a member has already taken? Cached sweeps
(rule = g 0). Rule-group members giving <= 4 at a punished round r in {4,5,8,9,12,13}, staying in the group
through r+3: rise of their mean contribution over r+1..r+3 above c_r, by dose at r and by the cumulative
punishment received before r (from either manager)."""
import sys, numpy as np, pandas as pd
rows = []
for f in sys.argv[1:]:
    z = np.load(f); c = z['c'].astype(np.float32); p = z['p'].astype(np.float32); g = z['g']
    cum = np.cumsum(p, -1) - p
    for r in (4, 5, 8, 9, 12, 13):
        sel = (g[..., r:r+4] == 0).all(-1) & (c[..., r] <= 4)
        rows.append(pd.DataFrame({'r': r, 'dose': p[..., r][sel], 'cum': cum[..., r][sel],
                                  'rise': c[..., r+1:r+4].mean(-1)[sel] - c[..., r][sel]}))
d = pd.concat(rows)
d['db'] = pd.cut(d.dose, [-1, 0, 10, 20, 30]); d['cb'] = pd.cut(d.cum, [-1, 20, 50, 90, 140, 1000])
pd.set_option('display.width', 200)
t = d.pivot_table(index='cb', columns='db', values='rise', aggfunc='mean', observed=True)
t['gain30'] = t.iloc[:, -1] - t.iloc[:, 0]
print(t.round(2)); print(d.pivot_table(index='cb', columns='db', values='rise', aggfunc='size', observed=True))
import statsmodels.formula.api as smf
r = smf.ols('rise ~ dose * cum + C(r)', d).fit(); print(r.params[['dose', 'cum', 'dose:cum']].round(5), r.bse[['dose', 'dose:cum']].round(5))
