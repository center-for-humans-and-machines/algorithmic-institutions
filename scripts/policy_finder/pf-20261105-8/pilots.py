"""Human pilots: punishment rule shape, response to punishment, end effects."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
d['valid'] = d.player_no_input == 0
key = ['experiment_name', 'episode_id', 'global_group_id']
d['n_rounds'] = d.groupby(key).round_number.transform('max') + 1
print(d.groupby(key).n_rounds.first().groupby('experiment_name').value_counts())
v = d[d.valid].copy()
v['gmean'] = v.groupby(key + ['round_number']).contribution.transform('mean')
v['gmax'] = v.groupby(key + ['round_number']).contribution.transform('max')
print('punishment by contribution, by experiment')
print(v.pivot_table(index=pd.cut(v.contribution, [-1, 0, 5, 10, 14, 19, 20]), columns='experiment_name', values='punishment', aggfunc='mean').round(2))
print('fraction punished')
print(v.assign(pp=v.punishment > 0).pivot_table(index=pd.cut(v.contribution, [-1, 0, 5, 10, 14, 19, 20]), columns='experiment_name', values='pp', aggfunc='mean').round(2))
print('mean contribution by round and n_rounds')
print(v.pivot_table(index='round_number', columns=['experiment_name', 'n_rounds'], values='contribution', aggfunc='mean').round(1))
d = d.sort_values(key + ['player_id', 'round_number'])
g = d.groupby(key + ['player_id'])
d['c_next'] = g.contribution.shift(-1); d['v_next'] = g.valid.shift(-1)
d['gmean'] = d.groupby(key + ['round_number']).contribution.transform('mean')
d['oth'] = (d.gmean * 4 - d.contribution) / 3
x = d[d.valid & (d.v_next == True)].copy()
x['dc'] = x.c_next - x.contribution
for e in ['random_1', 'trail_rounds_2']:
    y = x[x.experiment_name == e]
    print(e)
    print(y.pivot_table(index=pd.cut(y.contribution, [-1, 5, 10, 15, 20]), columns=pd.cut(y.punishment, [-1, 0, 3, 7, 12, 30]), values='dc', aggfunc='mean').round(2))
    print(y.pivot_table(index=pd.cut(y.contribution, [-1, 5, 10, 15, 20]), columns=pd.cut(y.punishment, [-1, 0, 3, 7, 12, 30]), values='dc', aggfunc='size'))
    X = np.column_stack([np.ones(len(y)), y.punishment, y.punishment * (y.contribution >= 15), y.contribution, y.oth, y.round_number])
    b = np.linalg.lstsq(X, y.dc, rcond=None)[0]
    print('dc ~ 1 + p + p*high + c + oth + t', np.round(b, 3))
