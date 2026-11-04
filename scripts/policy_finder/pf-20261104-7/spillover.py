"""v2 sweep (my group): does punishing others raise an unpunished member's contribution (herding/spillover)?
OLS of next-round change on own punishment bins, others' mean punishment, others' mean contribution, own contribution."""
import numpy as np
import pandas as pd
pd.set_option('display.width', 250)
cols = ['run', 'episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'agent_group']
d = pd.read_parquet('plots/simulation/policy_finder/pf-20261104-7-v2_sweep/per_round_p1of2.parquet', columns=cols, filters=[('episode', '<', 80)])
for col in ['run', 'participant_code']: d[col] = d[col].astype('category')
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number']).reset_index(drop=True)
g = d.groupby(['run', 'episode', 'participant_code'], sort=False, observed=True)
d['c_next'] = g.contribution.shift(-1); d['grp_next'] = g.agent_group.shift(-1)
gg = d.groupby(['run', 'episode', 'round_number', 'agent_group'], observed=True)
d['n'] = gg.contribution.transform('size')
d['oc'] = (gg.contribution.transform('sum') - d.contribution) / (d.n - 1).clip(lower=1)
d['op'] = (gg.punishment.transform('sum') - d.punishment) / (d.n - 1).clip(lower=1)
s = d[(d.agent_group == 0) & (d.grp_next == 0) & (d.round_number % 4 != 3) & (d.round_number < 23) & (d.n > 1)].copy()
s['dc'] = s.c_next - s.contribution
s['own0'] = (s.punishment == 0).astype(float)
# unpunished members only: dc vs others' mean punishment
u = s[s.punishment == 0]
u['opb'] = pd.cut(u.op, [-0.1, 0, 2, 5, 10, 30])
u['cb'] = pd.cut(u.contribution, [-1, 4, 9, 12, 15, 20])
print('unpunished members: next-round change by others mean punishment (rows) x own contribution')
print(u.pivot_table(index='opb', columns='cb', values='dc', aggfunc='mean', observed=True).round(2))
print(u.pivot_table(index='opb', columns='cb', values='dc', aggfunc='size', observed=True))
X = np.column_stack([np.ones(len(s)), s.contribution, s.oc, np.sqrt(s.punishment), s.op, s.round_number])
b, *_ = np.linalg.lstsq(X, s.dc.values, rcond=None)
res = s.dc.values - X @ b
se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var())
print('OLS dc ~ 1, c, others_c, sqrt(own p), others mean p, round')
print(pd.DataFrame({'b': b, 'se': se}, index=['1', 'c', 'oc', 'sqrt_p', 'op', 't']).round(4))
