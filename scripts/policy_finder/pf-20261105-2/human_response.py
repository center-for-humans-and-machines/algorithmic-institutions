"""How human contributions respond to punishment in the pilots."""
import sys
import numpy as np, pandas as pd
d = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
d['gid'] = d['global_group_id'].astype(str) + '_' + d['episode'].astype(str)
d = d.sort_values(['gid', 'participant_code', 'round_number'])
g = d.groupby(['gid', 'participant_code'])
d['next_c'] = g['contribution'].shift(-1)
d['next_valid'] = 1 - g['player_no_input'].shift(-1)
d['valid'] = 1 - d['player_no_input']
# others' mean contribution among valid
d['sum_c'] = d.groupby(['gid', 'round_number'])['contribution'].transform('sum')
d['n_valid'] = d.groupby(['gid', 'round_number'])['valid'].transform('sum')
d['oth_c'] = (d['sum_c'] - d['contribution']) / (d['n_valid'] - d['valid']).clip(lower=1)
d['dev'] = d['contribution'] - d['oth_c']
x = d[(d.valid == 1) & (d.next_valid == 1)].copy()
x['dc'] = x['next_c'] - x['contribution']
print('rounds', d.round_number.max() + 1)
print('mean c by round and experiment')
print(d[d.valid == 1].pivot_table(index='round_number', columns='experiment_name', values='contribution', aggfunc='mean').round(2))
print(d[d.valid == 1].pivot_table(index='round_number', columns='experiment_name', values='punishment', aggfunc='mean').round(2))
# binned response
x['pbin'] = pd.cut(x.punishment, [-1, 0, 3, 6, 10, 15, 30])
x['cbin'] = pd.cut(x.contribution, [-1, 4, 9, 14, 19, 20])
print('mean change in c next round by c bin x p bin (count)')
print(x.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
print(x.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='size'))
import numpy.linalg as la
X = np.column_stack([np.ones(len(x)), x.contribution, x.oth_c, x.punishment, np.minimum(x.punishment, 5), x.round_number])
b = la.lstsq(X, x.next_c, rcond=None)[0]
print('OLS next_c ~ 1, c, oth_c, p, min(p,5), round:', b.round(3))
for e in ['trail_rounds_2', 'random_1']:
    xe = x[x.experiment_name == e]
    X = np.column_stack([np.ones(len(xe)), xe.contribution, xe.oth_c, xe.punishment, xe.round_number])
    print(e, la.lstsq(X, xe.next_c, rcond=None)[0].round(3))
