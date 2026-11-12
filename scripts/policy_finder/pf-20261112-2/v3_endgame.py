"""v3: do fines after the last switch (rounds 20-23) pay? In v1's sweep t_stop is random,
and points with t_stop 19..23 play identically through round 19 given their other params.
Per point: change in my pool from rounds 16-19 to 20-23 (pool_me, and lead me-ah), vs the
number of endgame rounds fined (t_stop-19, 0..4) and the point's spend in rounds 16-19."""
import pandas as pd, numpy as np, json, numpy.linalg as la
x = pd.read_parquet('/tmp/v3_pr_pf-20261112-2.parquet')
pts = {p['name']: p['params'] for p in json.load(open('plots/simulation/policy_finder/pf-20261112-2_sweep/sweep.json'))['points']}
x['lead'] = x.pool_me - x.pool_ah
a = x.groupby(['pt', 'round_number'])[['pool_me', 'lead', 'p_me', 'c_me', 'n_me']].mean().reset_index()
def blk(lo, hi, col): return a[(a.round_number >= lo) & (a.round_number <= hi)].groupby('pt')[col].mean()
s = pd.DataFrame({'pool_late': blk(16, 19, 'pool_me'), 'pool_end': blk(20, 23, 'pool_me'), 'lead_late': blk(16, 19, 'lead'),
                  'lead_end': blk(20, 23, 'lead'), 'sp_late': blk(16, 19, 'p_me'), 'sp_end': blk(20, 23, 'p_me'), 'c_late': blk(16, 19, 'c_me'), 'c_end': blk(20, 23, 'c_me'), 'c23': blk(23, 23, 'c_me'), 'c20': blk(20, 20, 'c_me'), 'n_end': blk(20, 23, 'n_me')})
for k in ['thr', 'base', 'slope', 't_stop']: s[k] = [pts[i][k] for i in s.index]
s = s[s.t_stop >= 19].copy()
s['eg'] = s.t_stop - 19
s['d_pool'] = s.pool_end - s.pool_late; s['d_c'] = s.c_end - s.c_late; s['d_lead'] = s.lead_end - s.lead_late
print(len(s))
print(s.groupby('eg')[['d_pool', 'd_c', 'd_lead', 'sp_late', 'sp_end', 'n_end']].mean().round(2))
# regression: d_pool ~ sp_late + sp_late*eg (endgame fining intensity proxy)
X = np.column_stack([np.ones(len(s)), s.sp_late, s.sp_late * (s.eg > 0), s.sp_late * s.eg])
for o in ['d_pool', 'd_c', 'd_lead']:
    b, *_ = la.lstsq(X, s[o].values, rcond=None); r = s[o].values - X @ b
    se = np.sqrt(np.diag(la.inv(X.T @ X)) * r.var())
    print(o, 'coef [1, sp_late, sp_late*any_eg, sp_late*eg]', b.round(3), 'se', se.round(3))
