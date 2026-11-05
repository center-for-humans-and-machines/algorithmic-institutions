"""How do human contributions respond to punishment in the pilots?"""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
d = d.sort_values(['episode_id', 'participant_code', 'round_number'])
g = d.groupby(['episode_id', 'participant_code'])
d['c_next'] = g.contribution.shift(-1)
d['valid_next'] = 1 - g.player_no_input.shift(-1)
d['valid'] = 1 - d.player_no_input
grp = d[d.valid == 1].groupby(['episode_id', 'round_number']).contribution.agg(['sum', 'count'])
d = d.join(grp, on=['episode_id', 'round_number'])
d['others'] = (d['sum'] - d.contribution) / (d['count'] - 1)
d['dev'] = d.contribution - d.others
x = d[(d.valid == 1) & (d.valid_next == 1)].copy()
x['dc'] = x.c_next - x.contribution
x['pbin'] = pd.cut(x.punishment, [-1, 0, 2, 5, 10, 30])
x['cbin'] = pd.cut(x.contribution, [-1, 5, 10, 15, 19, 20])
print('round-by-round mean contribution by experiment')
print(d[d.valid == 1].pivot_table(index='round_number', columns='experiment_name', values='contribution').round(2).T)
print('\nnext-round change by punishment bin x own contribution bin (mean dc, n)')
print(x.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
print(x.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='count'))
print('\nsame, by experiment')
for e, xx in x.groupby('experiment_name'):
    print(e); print(xx.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
# regression dc ~ p + c + others
import numpy.linalg as la
X = np.c_[np.ones(len(x)), x.punishment, x.contribution, x.others, x.punishment * (x.contribution < 20)]
b = la.lstsq(X, x.dc, rcond=None)[0]
print('\nOLS dc ~ 1 + p + c + others + p*(c<20):', b.round(3))
# how do human managers punish
h = d[(d.experiment_name == 'trail_rounds_2') & (d.valid == 1) & (d.manager_no_input == 0)]
print('\nhuman manager punishment by own contribution bin')
print(h.groupby(pd.cut(h.contribution, [-1, 5, 10, 15, 19, 20])).punishment.agg(['mean', 'count', lambda s: (s > 0).mean()]))
print('human manager punishment by deviation from others')
print(h.groupby(pd.cut(h.dev, [-21, -10, -5, -2, 2, 21])).punishment.agg(['mean', 'count']))
