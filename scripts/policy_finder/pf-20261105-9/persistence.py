"""Humanlike sim: effect of punishment at t on own contribution at t+k, and on others at t+1."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d = d[d.run.str.contains('humanlike')].copy()
d['pid'] = d.participant_code.str.split('_').str[0].astype(int)
d = d.sort_values(['episode', 'pid', 'round_number'])
d['c'] = np.where(d.contribution_valid, d.contribution, 0)
d['v'] = d.contribution_valid.astype(int)
g = d.groupby(['episode', 'round_number'])
d['n'] = g.v.transform('sum')
d['oth_c'] = (g.c.transform('sum') - d.c) / (d.n - d.v).clip(lower=1)
d['oth_p'] = (g.punishment.transform('sum') - d.punishment) / (d.n - d.v).clip(lower=1)
k = d.groupby(['episode', 'pid'])
d['c_prev'] = k.c.shift(1)
d['p_prev'] = k.punishment.shift(1)
for lag in range(1, 7):
    d[f'c{lag}'] = k.c.shift(-lag); d[f'v{lag}'] = k.v.shift(-lag)
d['low'] = (d.c < 10).astype(int)
base = d[(d.v == 1) & d.c_prev.notna()]
for lag in range(1, 7):
    x = base[base[f'v{lag}'] == 1]
    m = smf.ols(f'c{lag} ~ c + c_prev + oth_c + p_prev + oth_p + punishment + punishment:low + C(round_number)', x).fit()
    print(lag, {k_: round(v_, 3) for k_, v_ in m.params.items() if 'round' not in k_ and k_ != 'Intercept'})
