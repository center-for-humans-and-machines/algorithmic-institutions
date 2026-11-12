"""v3: design-level value of a point of spend by block. Per design point (both seats),
lead = mean over rounds of (my pool - ah's pool); regress on my group's total spend per
round in each block (0-3, 4-7, ..., 20-23). In v1 the blocks' spends differ through t_stop,
thr, base, slope; in v2 through n_dec and w. Coefficient = lead change per point of spend
in that block (spend is itself counted in the lead with weight 1/24 per round)."""
import pandas as pd, numpy as np, numpy.linalg as la, sys
for v in ['pf-20261112-2', 'pf-20261112-2-v2']:
    x = pd.read_parquet(f'/tmp/v3_pr_{v}.parquet')
    x['lead'] = x.pool_me - x.pool_ah
    x['b'] = x.round_number // 4
    a = x.groupby(['pt', 'b'])[['p_me', 'lead', 'n_me', 'c_me']].mean().unstack('b')
    S = a['p_me'].values  # mean spend per round in block (sum over my members)
    lead = a['lead'].values.mean(1)
    keep = S.std(0) > 1e-6
    X = np.column_stack([np.ones(len(S)), S[:, keep] * 4 / 24])  # coefficient per point of total spend
    b, *_ = la.lstsq(X, lead, rcond=None); r = lead - X @ b
    se = np.sqrt(np.diag(la.inv(X.T @ X)) * r.var() * len(r) / (len(r) - X.shape[1]))
    print(v, 'blocks', np.nonzero(keep)[0], '\n  lead change per point of total spend:', b[1:].round(3), '\n  se', se[1:].round(3), ' intercept', b[0].round(2))
    print('  corr of block spends:\n', np.corrcoef(S[:, keep].T).round(2))
    # members at end by spend
    n = a['n_me'].values.mean(1)
    b2, *_ = la.lstsq(X, n, rcond=None)
    print('  members change per point of spend', (b2[1:] * 100).round(3), 'x1e-2')
