"""Persistence of the punishment effect on contribution in the reference sims (ah groups, same group throughout)."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('ah_vs_zero|zero_vs_ah')]
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
g = d.groupby(['run', 'episode', 'participant_code'])
d['gmean'] = d.groupby(['run', 'episode', 'round_number', 'agent_group']).contribution.transform('mean')
for k in [1, 2, 3, 4, 6, 8]:
    d[f'c{k}'] = g.contribution.shift(-k)
    d[f'g{k}'] = g.agent_group.shift(-k)
x = d[(d.punishment.notna())]
for k in [1, 2, 3, 4, 6, 8]:
    y = x[(x[f'g{k}'] == x.agent_group) & x[f'c{k}'].notna() & (x.round_number >= 4)]
    y = y.assign(low=(y.contribution <= 5).astype(float))
    m = smf.ols(f'c{k} ~ contribution + gmean + punishment + punishment:low + C(round_number)', y).fit()
    print(k, 'punish coef %.3f  extra for c<=5 %.3f  n %d' % (m.params['punishment'], m.params['punishment:low'], len(y)))
