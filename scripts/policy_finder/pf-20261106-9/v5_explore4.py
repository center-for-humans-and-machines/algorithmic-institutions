# v5: game-level value of our round-0 and round-1 punishment in v4's sweep, per opponent.
# OLS of the game's pool (mean per round), our members at round 4, and our mean contribution in
# rounds 6-23 on our total punishment in round 0 (P0) and round 1 (P1), controlling for our
# round-0 contributions (sum, sd, min) and the other group's round-0 contributions (first run also
# controlled for round-1 contributions, a post-treatment control for P0; dropped).
import numpy as np, statsmodels.api as sm
d = np.load('/tmp/v4sweep.npz')
c, p, v, g, ours, opp = d['c'].astype(float), d['p'].astype(float), d['v'], d['g'], d['ours'], d['opp']
M = g == ours[:, None, None]
pool = (1.6 * (c * M).sum(1) - (p * M).sum(1)).mean(1)
P0 = (p[:, :, 0] * M[:, :, 0]).sum(1); P1 = (p[:, :, 1] * M[:, :, 1]).sum(1)
Q0 = (p[:, :, 0] * ~M[:, :, 0]).sum(1); Q1 = (p[:, :, 1] * ~M[:, :, 1]).sum(1)
n4 = M[:, :, 4].sum(1)
cl = np.where(M[:, :, 6:], c[:, :, 6:], np.nan); clate = np.nanmean(cl.reshape(len(c), -1), 1)
c0 = np.where(M[:, :, 0], c[:, :, 0], np.nan); c1 = np.where(M[:, :, 1], c[:, :, 1], np.nan)
ctrl = np.column_stack([np.nansum(c0, 1), np.nanstd(c0, 1), np.nanmin(c0, 1), np.where(~M[:, :, 0], c[:, :, 0], 0).sum(1)])
for o in range(1, 5):
    s = opp == o
    print(f'== opponent {o}: mean P0 {P0[s].mean():.1f} P1 {P1[s].mean():.1f}  other Q0 {Q0[s].mean():.1f} Q1 {Q1[s].mean():.1f}')
    X = sm.add_constant(np.column_stack([P0[s], P1[s], ctrl[s]]))
    for nm, y in [('pool', pool[s]), ('members r4', n4[s]), ('c late', clate[s])]:
        ok = ~np.isnan(y)
        r = sm.OLS(y[ok], X[ok]).fit()
        print(f'  {nm:10s} P0 {r.params[1]:+.4f} ({r.bse[1]:.4f})  P1 {r.params[2]:+.4f} ({r.bse[2]:.4f})')
# pooled with interaction of P1 with the other's round-0 punishment
X = sm.add_constant(np.column_stack([P0, P1, P1 * Q0 / 10, Q0, ctrl]))
r = sm.OLS(pool, X).fit()
print('pooled pool: P0 %+.4f P1 %+.4f P1xQ0/10 %+.4f Q0 %+.4f' % tuple(r.params[1:5]), 'se', np.round(r.bse[1:5], 4))
X = sm.add_constant(np.column_stack([P0, P0 * Q0 / 10, Q0, P1, P1 * Q0 / 10, ctrl]))
r = sm.OLS(pool, X).fit()
print('pooled pool: P0 %+.4f P0xQ0/10 %+.4f Q0 %+.4f P1 %+.4f P1xQ0/10 %+.4f' % tuple(r.params[1:6]), 'se', np.round(r.bse[1:6], 4))
