"""v6: migration at each decision in v5's sweep (part 1), group 0 = ours. Per game and
decision round r (3,7,..,19): our size before (round r) and after (round r+1), the per-member
pool share of each group in round r and over the block, our punishment in the block and at r.
Saves v6_migr.parquet; prints transition tables and a regression of the size change."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
B = 'plots/simulation/policy_finder/pf-20261104-5-v5_sweep/'
A = load(B + 'per_round_p1of2.parquet')
c, p, g = A['c'], A['p'], A['g'] == 0
n = g.sum(2).astype(float)
P0 = (1.6 * c * g).sum(2) - (p * g).sum(2)
P1 = (1.6 * c * ~g).sum(2) - (p * ~g).sum(2)
s0 = np.where(n > 0, P0 / np.maximum(n, 1), np.nan)
s1 = np.where(n < 8, P1 / np.maximum(8 - n, 1), np.nan)
pay = 20 - c + np.where(g, s0[..., None], s1[..., None]) - p   # player payoff
pay0 = np.where(n > 0, (pay * g).sum(2) / np.maximum(n, 1), np.nan)
pay1 = np.where(n < 8, (pay * ~g).sum(2) / np.maximum(8 - n, 1), np.nan)
rows = []
for r in [3, 7, 11, 15, 19]:
    sl = slice(r - 3, r + 1)
    rows.append(pd.DataFrame(dict(
        game=np.arange(len(n)), r=r, n_before=n[:, r], n_after=n[:, r + 1],
        s0=s0[:, r], s1=s1[:, r], s0b=np.nanmean(s0[:, sl], 1), s1b=np.nanmean(s1[:, sl], 1),
        pay0=pay0[:, r], pay1=pay1[:, r], pay0b=np.nanmean(pay0[:, sl], 1), pay1b=np.nanmean(pay1[:, sl], 1),
        pblk=(p * g)[:, sl].sum((1, 2)), pdec=(p * g)[:, r].sum(1),
        pblk1=(p * ~g)[:, sl].sum((1, 2)),
        cm0=(c * g)[:, sl].sum((1, 2)) / np.maximum(g[:, sl].sum((1, 2)), 1),
        cm1=(c * ~g)[:, sl].sum((1, 2)) / np.maximum((~g)[:, sl].sum((1, 2)), 1))))
M = pd.concat(rows, ignore_index=True)
M['run'] = np.array([x.split('managed by ')[1].split('_vs_')[0] for x in A['run']])[M.game]
M.to_parquet('scripts/policy_finder/pf-20261104-5/v6_migr.parquet')
M['dn'] = M.n_after - M.n_before
print(pd.crosstab(M[M.r == 3].n_before, M[M.r == 3].n_after))
for r in [7, 11]:
    print('r', r); print((pd.crosstab(M[M.r == r].n_before, M[M.r == r].n_after, normalize='index') * 100).round(0))
mid = M[(M.n_before >= 1) & (M.n_before <= 7)].copy()
mid['gap_pay'] = mid.pay0b - mid.pay1b
mid['gap_s'] = mid.s0b - mid.s1b
mid['gap_pay_r'] = mid.pay0 - mid.pay1
print(mid.groupby([pd.cut(mid.gap_pay, [-100, -15, -8, -4, -1, 1, 4, 8, 15, 100])], observed=True).dn.agg(['mean', 'size']).round(2))
print(mid.groupby([pd.cut(mid.gap_pay_r, [-100, -15, -8, -4, -1, 1, 4, 8, 15, 100])], observed=True).dn.agg(['mean', 'size']).round(2))
from numpy.linalg import lstsq
for r in [3, 7, 11]:
    s = mid[mid.r == r]
    for cols in [['gap_pay'], ['gap_pay_r'], ['gap_s'], ['gap_pay', 'pblk', 'pdec', 'pblk1', 'n_before'], ['gap_s', 'pblk', 'pdec', 'pblk1', 'n_before', 'cm0', 'cm1']]:
        X = np.column_stack([np.ones(len(s))] + [s[c_] for c_ in cols])
        b, res, *_ = lstsq(X, s.dn.values, rcond=None)
        r2 = 1 - res[0] / ((s.dn - s.dn.mean()) ** 2).sum()
        print(r, cols, b.round(4), 'R2', round(r2, 3))
