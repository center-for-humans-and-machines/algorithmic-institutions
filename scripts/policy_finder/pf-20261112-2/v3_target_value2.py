"""v3: robustness of v3_target_value: coarser bands (0, 1-9, 10+), rounds 4-19 pooled;
correlations between band spends; and a bootstrap over design points."""
import pandas as pd, numpy as np, numpy.linalg as la
s = pd.read_csv('scripts/policy_finder/pf-20261112-2/v3_target_value.csv', index_col=0)
s['early'] = s.filter(like='r0_3').sum(1); s['end'] = s.filter(like='r20').sum(1)
s['mid0'] = s.r4_19_c0; s['mid19'] = s.r4_19_c1_4 + s.r4_19_c5_9; s['mid10'] = s.r4_19_c10p
print(s[['early', 'mid0', 'mid19', 'mid10', 'end']].corr().round(2))
def fit(cols, d):
    X = np.column_stack([np.ones(len(d))] + [d[c] / 24 for c in cols])
    b, *_ = la.lstsq(X, d.lead.values, rcond=None); r = d.lead.values - X @ b
    se = np.sqrt(np.diag(la.inv(X.T @ X)) * r.var() * len(r) / (len(r) - X.shape[1]))
    return b, se
for cols in [['early', 'mid0', 'mid19', 'mid10', 'end'], ['early', 'r4_19_c0', 'r4_19_c1_4', 'r4_19_c5_9', 'r4_19_c10p', 'end']]:
    b, se = fit(cols, s)
    print(' '.join(f'{c}={bb:.2f}+-{ss:.2f}' for c, bb, ss in zip(['const'] + cols, b, se)))
rng = np.random.default_rng(0)
B = np.array([fit(['early', 'mid0', 'mid19', 'mid10', 'end'], s.iloc[rng.integers(0, len(s), len(s))])[0] for _ in range(500)])
print('bootstrap mean', B.mean(0).round(2), 'P(mid19 > -1)', (B[:, 3] > -1).mean(), 'P(mid19 > 0)', (B[:, 3] > 0).mean())
