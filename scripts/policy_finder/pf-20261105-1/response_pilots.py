import pandas as pd, numpy as np
df = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
df['valid'] = 1-df.player_no_input
df = df.sort_values(['episode_id','participant_code','round_number'])
g = df.groupby(['episode_id','participant_code'])
df['c_next'] = g.contribution.shift(-1)
df['v_next'] = g.valid.shift(-1)
# others' mean contribution this round
grp = df[df.valid==1].groupby(['episode_id','round_number']).contribution.agg(['sum','count'])
df = df.join(grp, on=['episode_id','round_number'])
df['others'] = (df['sum']-df.contribution*df.valid)/(df['count']-df.valid)
d = df[(df.valid==1)&(df.v_next==1)].copy()
d['dc'] = d.c_next - d.contribution
d['pbin'] = pd.cut(d.punishment,[-1,0,2,5,10,30])
d['cbin'] = pd.cut(d.contribution,[-1,4,9,14,19,20])
for exp in ['random_1','trail_rounds_2','all']:
    x = d if exp=='all' else d[d.experiment_name==exp]
    print('\n==',exp,'mean next-round change in contribution, by own contribution bin x punishment bin')
    print(x.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean',observed=False).round(2))
    print(x.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='count',observed=False))
# regression
import numpy.linalg as la
x = d.copy()
x['gap'] = x.others - x.contribution
X = np.c_[np.ones(len(x)), x.contribution, x.gap, x.punishment, np.minimum(x.punishment,5), x.round_number]
b = la.lstsq(X, x.dc, rcond=None)[0]
print('\nOLS dc ~ 1 + c + (others-c) + p + min(p,5) + round:', b.round(3))
