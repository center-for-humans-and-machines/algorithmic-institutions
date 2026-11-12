"""ah-managed groups in ref sims: lasting effect of a punishment on a member's contribution,
and the effect on other members (bystanders). Regressions within group, no switch in window."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['agent_group'] = d.agent_group.astype(int)
d = d[d.run.isin(['ah group_switching managed by zero_vs_ah', 'ah group_switching managed by ah_vs_zero', 'ah group_switching managed by ah_vs_ah'])].copy()
d['g'] = d.run + d.episode.astype(str)
d = d.sort_values(['g', 'participant_code', 'round_number'])
gs = d.groupby(['g', 'round_number', 'agent_group']).agg(sp=('punishment', 'sum'), n=('punishment', 'size'), sc=('contribution', 'sum')).reset_index()
d = d.merge(gs, on=['g', 'round_number', 'agent_group'])
d['op'] = (d.sp - d.punishment) / (d.n - 1).clip(lower=1)   # others' mean punishment
d['oc'] = (d.sc - d.contribution) / (d.n - 1).clip(lower=1)
d = d.sort_values(['g', 'participant_code', 'round_number'])
gg = d.groupby(['g', 'participant_code'])
for k in (1, 2, 3):
    d[f'c{k}'] = gg.contribution.shift(-k)
    d[f'grp{k}'] = gg.agent_group.shift(-k)
d['low'] = (d.contribution <= 5).astype(int)
# ah-managed: zero_vs_ah group 1, ah_vs_zero group 0, ah_vs_ah both
mz = (d.run.str.contains('zero_vs_ah') & (d.agent_group == 0)) | (d.run.str.contains('ah_vs_zero') & (d.agent_group == 1))
x = d[~mz & d.contribution_valid.astype(bool)]
for k in (1, 2, 3):
    y = x[(x[f'grp{k}'] == x.agent_group) & (x.round_number % 4 + k <= 3)].copy()  # no switch decision inside window
    y['dc'] = y[f'c{k}'] - y.contribution
    m = smf.ols('dc ~ contribution + oc + punishment*low + op + round_number', y).fit()
    print(k, 'n', len(y), m.params.round(3).to_dict())

print('longer horizon, player in same group at t+k (switches allowed in between), low givers only')
for k in (1, 2, 4, 6, 8, 12):
    d[f'ck'] = gg.contribution.shift(-k); d['gk'] = gg.agent_group.shift(-k)
    x = d[~mz & d.contribution_valid.astype(bool)]
    y = x[(x.gk == x.agent_group) & (x.contribution <= 5)].copy()
    y['dc'] = y.ck - y.contribution
    m = smf.ols('dc ~ contribution + oc + punishment + op + round_number', y).fit()
    print(k, 'n', len(y), 'punishment %.3f (se %.3f)' % (m.params.punishment, m.bse.punishment), 'oc %.3f' % m.params.oc)
