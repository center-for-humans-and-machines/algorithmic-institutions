"""Reference sims: contribution path k rounds after a punishment at round t, by own c_t (players under ah).
Compares punished (p_t >= thr) vs unpunished, conditioning on c_t and previous c (t-1)."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
key = ['run', 'episode', 'participant_code']
for k in range(1, 5):
    d[f'c{k}'] = d.groupby(key).contribution.shift(-k)
d['cprev'] = d.groupby(key).contribution.shift(1)
d['g4'] = d.groupby(key).agent_group.shift(-4)
v = d[(d.contribution_valid == 1) & (d.round_number.between(1, 18)) & d.cprev.notna()].copy()
v['cbin'] = pd.cut(v.contribution, [-1, 0, 4, 8, 12, 20])
v['pun'] = np.select([v.punishment == 0, v.punishment <= 5, v.punishment > 5], ['0', '1-5', '>5'])
for k in range(1, 5):
    v[f'd{k}'] = v[f'c{k}'] - v.contribution
t = v.groupby(['cbin', 'pun'], observed=True)[['d1', 'd2', 'd3', 'd4']].mean().round(2)
t['n'] = v.groupby(['cbin', 'pun'], observed=True).size()
print(t)
# regression controlling for cprev and round
import statsmodels.formula.api as smf
for k in (1, 2, 3, 4):
    w = v[v[f'c{k}'].notna()]
    m = smf.ols(f'd{k} ~ punishment*contribution + cprev + round_number', data=w).fit()
    print(k, m.params.round(3).to_dict())
