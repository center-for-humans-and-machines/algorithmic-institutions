"""Human data: managers' punishment by contribution and round; contribution response; switching;
per-game pool lead vs punishment difference between managers."""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
print('games', d.groupby(['session', 'episode']).ngroups, 'rows', len(d))
d['valid'] = 1 - d.player_no_input
v = d[d.valid == 1].copy()
v['cbin'] = pd.cut(v.contribution, [-1, 0, 4, 8, 12, 16, 20])
print('== human managers: mean punishment & P(p>0) by contribution')
print(v.groupby('cbin', observed=True).punishment.agg(['mean', lambda s: (s > 0).mean(), 'size']).round(2))
print('== by round: mean c, mean p')
print(v.groupby('round_number')[['contribution', 'punishment']].mean().round(2).T.to_string())
d = d.sort_values(['session', 'episode', 'participant_code', 'round_number'])
key = ['session', 'episode', 'participant_code']
d['c1'] = d.groupby(key).contribution.shift(-1); d['v1'] = d.groupby(key).valid.shift(-1)
d['gn'] = d.groupby(key).group_id.shift(-1)
w = d[(d.valid == 1) & (d.v1 == 1) & (d.round_number % 4 != 3)].copy()
w['dc'] = w.c1 - w.contribution
w['cbin'] = pd.cut(w.contribution, [-1, 0, 4, 8, 12, 20])
w['pun'] = np.select([w.punishment == 0, w.punishment <= 5, w.punishment > 5], ['0', '1-5', '>5'])
print('== humans: next-round change by c and p')
print(w.pivot_table(index='cbin', columns='pun', values='dc', aggfunc='mean', observed=True).round(2))
print(w.pivot_table(index='cbin', columns='pun', values='dc', aggfunc='size', observed=True))
# switching
d['blk'] = d.round_number // 4
b = d[d.round_number < 20].groupby(key + ['blk']).agg(p=('punishment', 'sum'), c=('contribution', 'mean'), g=('group_id', 'last'), gn=('gn', 'last')).reset_index()
b['sw'] = (b.g != b.gn).astype(float)
b['pbin'] = pd.cut(b.p, [-1, 0, 4, 10, 20, 40, 200])
print('== humans switch rate by block punishment')
print(b.groupby('pbin', observed=True).sw.agg(['mean', 'size']).round(3))
# per game: total pool lead and punishment difference
d['pool'] = 1.6 * d.contribution - d.punishment
g = d.groupby(['session', 'episode', 'round_number', 'group_id']).agg(pool=('pool', 'sum'), p=('punishment', 'sum'), n=('pool', 'size')).unstack(fill_value=0)
ep = g.groupby(['session', 'episode']).mean()
lead = ep['pool'][0] - ep['pool'][1]; dp = ep['p'][0] - ep['p'][1]; dn = ep['n'][0] - ep['n'][1]
print('== per game: corr(lead, punishment diff) %.2f, corr(lead, size diff) %.2f, corr(size diff, pun diff) %.2f' % (np.corrcoef(lead, dp)[0, 1], np.corrcoef(lead, dn)[0, 1], np.corrcoef(dn, dp)[0, 1]))
import statsmodels.api as sm
print(sm.OLS(lead.values, sm.add_constant(dp.values)).fit().params, 'lead per point of group punishment per round')
