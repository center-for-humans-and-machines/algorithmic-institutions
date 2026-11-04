"""Reference sims: do the artificial humans respond to punishment like humans? (contribution change, switching)"""
import pandas as pd, numpy as np
pd.set_option('display.width', 200)
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('ah_vs_zero|zero_vs_ah')].copy()
d['mgr'] = np.where(d.run.str.contains('ah_vs_zero'), np.where(d.agent_group == 0, 'ah', 'zero'), np.where(d.agent_group == 1, 'ah', 'zero'))
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
gp = d.groupby(['run', 'episode', 'participant_code'])
for k in (1, 3):
    d[f'c{k}'] = gp.contribution.shift(-k); d[f'g{k}'] = gp.agent_group.shift(-k)
x = d[d.contribution_valid & (d.round_number % 4 != 3) & (d.g1 == d.agent_group)].copy()
x['cb'] = pd.cut(x.contribution, [-1, 0, 4, 9, 14, 19, 20]); x['pb'] = pd.cut(x.punishment, [-1, 0, 2, 4, 7, 12, 30])
x['dc'] = x.c1 - x.contribution
print('sim: next-round change'); print(x.pivot_table(index='pb', columns='cb', values='dc', aggfunc='mean', observed=True).round(2))
print(x.pivot_table(index='pb', columns='cb', values='dc', aggfunc='count', observed=True))
y = d[d.contribution_valid & (d.round_number <= 18) & (d.contribution <= 9) & (d.g3 == d.agent_group)]
print('t+3 gain low contributors', (y.c3 - y.contribution).groupby(pd.cut(y.punishment, [-1, 0, 4, 30]), observed=True).agg(['mean', 'count']).round(2))
# switching
s = d[(d.round_number % 4 == 3) & (d.round_number < 23)].copy()
s['sw'] = (s.g1 != s.agent_group).astype(float)
s['p4'] = gp.punishment.transform(lambda v: v.rolling(4, 1).sum()).loc[s.index]
print('sim switch rate by manager', s.groupby('mgr').sw.mean().round(3).to_dict())
print(s[s.mgr == 'ah'].groupby(pd.cut(s.p4, [-1, 0, 3, 8, 20, 200]), observed=True).sw.agg(['mean', 'count']).round(3))
print('punishment in the switch round vs earlier (ah group)')
rows = []
for r in (3, 7, 11, 15, 19):
    a = d[d.round_number.between(r - 3, r)].pivot_table(index=['run', 'episode', 'participant_code'], columns='round_number', values='punishment')
    s2 = s[s.round_number == r].set_index(['run', 'episode', 'participant_code'])
    s2 = s2[s2.mgr == 'ah']
    rows.append(pd.DataFrame({'sw': s2.sw, 'plast': a[r].reindex(s2.index), 'pprev': a[[r - 1, r - 2, r - 3]].sum(1).reindex(s2.index), 'c': s2.contribution}))
import statsmodels.formula.api as smf
m = smf.logit('sw ~ plast + pprev + c', pd.concat(rows).dropna()).fit(disp=0); print(m.params.round(3).to_dict(), m.bse.round(3).to_dict())
