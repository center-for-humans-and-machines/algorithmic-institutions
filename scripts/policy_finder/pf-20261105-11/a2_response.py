import pandas as pd, numpy as np
pd.set_option('display.width',200)
d = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
d = d.sort_values(['episode_id','participant_code','round_number'])
g = d.groupby(['episode_id','participant_code'])
d['c_next'] = g.contribution.shift(-1); d['to_next']=g.player_no_input.shift(-1)
# group mean of others
gm = d.groupby(['episode_id','round_number']).contribution.transform('sum')
nv = d.groupby(['episode_id','round_number']).player_no_input.transform(lambda x:(1-x).sum())
d['others'] = (gm - d.contribution)/(nv-1+ (d.player_no_input)).clip(lower=1)
x = d[(d.player_no_input==0)&(d.to_next==0)].copy()
x['dc'] = x.c_next - x.contribution
x['cb'] = pd.cut(x.contribution,[-1,4,9,14,19,20])
x['pb'] = pd.cut(x.punishment,[-1,0,2,5,10,30])
for e in ['trail_rounds_2','random_1']:
    print(e)
    print(x[x.experiment_name==e].pivot_table(index='cb',columns='pb',values='dc',aggfunc=['mean','size'],observed=True).round(2))
# regression dc ~ p + c + others-c, by experiment
import numpy.linalg as la
for e in ['trail_rounds_2','random_1', None]:
    z = x if e is None else x[x.experiment_name==e]
    X = np.c_[np.ones(len(z)), z.punishment, z.contribution, z.others - z.contribution, (z.punishment>0)]
    b = la.lstsq(X, z.dc.values, rcond=None)[0]
    print(e, 'dc ~ 1, p, c, others-c, p>0:', b.round(3))
# Low contributors only (c<15)
z = x[x.contribution<15]
X = np.c_[np.ones(len(z)), z.punishment, z.contribution, z.others - z.contribution]
print('c<15', la.lstsq(X, z.dc.values, rcond=None)[0].round(3))
