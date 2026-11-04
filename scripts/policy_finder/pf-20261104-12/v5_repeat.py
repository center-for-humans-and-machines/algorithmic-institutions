"""v5: do stubborn free riders still answer punishment? Cached sweeps (rule = g 0). Members in the rule's group
rounds 0-11 who gave <= cmax at round 0 and were punished >= 15 at round 0. Type by round-1 response:
'stubborn' if c1 <= cmax, else 'moved'. Then by round-4 contribution band, OLS of their mean contribution in
rounds 6-11 on the round-4 and round-5 punishment (variation mostly across design points), with c0..c4."""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
cmax = 2; rows = []
for f in sys.argv[1:]:
    z = np.load(f); c = z['c'].astype(np.float32); p = z['p'].astype(np.float32); g = z['g']
    keep = (g[..., :12] == 0).all(-1) & (c[..., 0] <= cmax) & (p[..., 0] >= 15)
    pi, ei, ii = np.nonzero(keep)
    d = pd.DataFrame({'f': f, 'pt': pi, **{f'c{r}': c[pi, ei, ii, r] for r in range(6)}, 'p1': p[pi, ei, ii, 1],
                      'p4': p[pi, ei, ii, 4], 'p5': p[pi, ei, ii, 5], 'p8': p[pi, ei, ii, 8],
                      'late': c[pi, ei, ii, 6:12].mean(-1), 'late2': c[pi, ei, ii, 12:16].mean(-1) if True else 0})
    rows.append(d)
d = pd.concat(rows)
d['type'] = np.where(d.c1 <= cmax, 'stubborn', 'moved')
d['type3'] = np.where((d.c1 <= cmax) & (d.c4 <= cmax), 'stubborn2', np.where(d.c4 <= cmax, 'relapsed', 'up'))
print(d.groupby('type3')[['c0','c1','c2','c3','c4','p1','p4','p5','late']].mean().round(2).assign(N=d.groupby('type3').size()))
for t, x in d.groupby('type3'):
    r = smf.ols('late ~ p4 + p5 + c1 + c2 + c3 + c4 + p1', x).fit()
    print(t, 'per point p4', round(r.params.p4, 3), '(', round(r.bse.p4, 3), ') p5', round(r.params.p5, 3), '(', round(r.bse.p5, 3), ')')
    x = x.assign(p4b=pd.cut(x.p4, [-1, 0, 10, 20, 29, 30]))
    print(x.groupby('p4b', observed=True).late.agg(['mean', 'size']).round(2).T.to_string())
