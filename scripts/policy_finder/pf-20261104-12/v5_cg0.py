"""v5: episode level, cached sweeps (rule = g 0): net members at decision 3 (size in round 4 minus 4)
by the rule group's round-0 and round-1 per-member common good (1.6 mean c - mean p), with the
rounds 2-3 common good and ah's block-0 common good held in bins; is the member cost of a low
early common good linear or saturating?"""
import sys, numpy as np, pandas as pd
rows = []
for f in sys.argv[1:]:
    z = np.load(f); c = z['c'].astype(np.float32); p = z['p'].astype(np.float32); g = z['g']
    m0 = (g == 0); m1 = (g == 1)
    def cg(m, r):
        n = m[..., r].sum(-1); return ((1.6*c[..., r] - p[..., r])*m[..., r]).sum(-1) / np.maximum(n, 1)
    df = pd.DataFrame({'cg0': cg(m0, 0).ravel(), 'cg1': cg(m0, 1).ravel(), 'cg23': ((cg(m0, 2)+cg(m0, 3))/2).ravel(),
                       'ah03': np.mean([cg(m1, r) for r in range(4)], 0).ravel(),
                       'dn': (m0[..., 4].sum(-1) - 4).ravel(),
                       'c_late': ((c[..., 12:]*m0[..., 12:]).sum((-1, -2)) / np.maximum(m0[..., 12:].sum((-1, -2)), 1)).ravel(),
                       'pool': ((1.6*c - p)*m0).sum(-2).mean(-1).ravel()})
    rows.append(df)
d = pd.concat(rows)
import statsmodels.formula.api as smf
d['cg0b'] = pd.cut(d.cg0, [-40, -10, -5, 0, 5, 10, 15, 20, 33])
print(d.groupby('cg0b', observed=True)[['dn','cg1','cg23','ah03','c_late','pool']].mean().assign(n=d.groupby('cg0b', observed=True).size()).round(2))
r = smf.ols('dn ~ C(cg0b) + cg1 + cg23 + ah03', d).fit(); print(r.params.round(3))
d['cg1b'] = pd.cut(d.cg1, [-40, -10, -5, 0, 5, 10, 15, 20, 33])
r = smf.ols('dn ~ cg0 + C(cg1b) + cg23 + ah03', d).fit(); print(r.params.round(3))
