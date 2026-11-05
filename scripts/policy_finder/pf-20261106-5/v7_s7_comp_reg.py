# v7: from /tmp/v7_comp.csv (v7_s6): within round-0 composition, the effect of how many were hit, controlling for c0 and opponent, per version
import pandas as pd, numpy as np, statsmodels.formula.api as smf
pd.set_option('display.width', 250)
a = pd.read_csv('/tmp/v7_comp.csv')
a['depth'] = a.p0 / a.nhit.clip(lower=1)
a = a[((a.nhit == 0) | (a.depth >= 12)) & (a.n == 4)]
a['k'] = a.nhit.clip(upper=4).astype(str)
for v in sorted(a.v.unique()):
    for nl in [1, 2, 3]:
        s = a[(a.v == v) & (a.nlow == nl)]
        f = smf.ols('pool ~ C(k, Treatment("%d")) + c0 + C(opp)' % nl, data=s).fit()
        co = {k.split('[T.')[1][:-1]: (round(f.params[k], 1), round(f.bse[k], 1)) for k in f.params.index if 'C(k' in k}
        print(v, 'nlow', nl, 'ref k=%d' % nl, co, 'n', len(s))
