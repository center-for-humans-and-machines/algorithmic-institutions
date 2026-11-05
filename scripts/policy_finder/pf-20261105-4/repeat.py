"""Do clones keep responding to repeated punishment? humanlike run."""
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
sim = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d = sim[sim.run.str.endswith('humanlike')].sort_values(['participant_code', 'round_number']).copy()
d.loc[~d.contribution_valid, 'contribution'] = np.nan
g = d.groupby('participant_code')
d['c1'] = g.contribution.shift(-1)
d['npun'] = g.punishment.transform(lambda s: (s > 0).cumsum().shift(1, fill_value=0))
d['cmean_prev'] = g.contribution.transform(lambda s: s.expanding().mean().shift(1))
x = d[(d.contribution <= 10)].dropna(subset=['c1'])
x = x.assign(dc=x.c1 - x.contribution, pun=x.punishment > 2)
print(x.pivot_table(index=pd.cut(x.npun, [-1, 0, 2, 5, 10, 24]), columns='pun', values='dc').round(2))
print(x.pivot_table(index=pd.cut(x.npun, [-1, 0, 2, 5, 10, 24]), columns='pun', values='dc', aggfunc='count'))
# share of players who are persistently low
m = d.groupby('participant_code').contribution.mean()
print('player mean contribution distribution (humanlike):', np.histogram(m, [0, 3, 6, 10, 15, 21])[0])
z = sim[sim.run.str.endswith('zero')].groupby('participant_code').contribution.mean()
print('player mean contribution distribution (zero):', np.histogram(z, [0, 3, 6, 10, 15, 21])[0])
