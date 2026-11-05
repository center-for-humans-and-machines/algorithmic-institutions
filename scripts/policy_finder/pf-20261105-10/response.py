"""How next-round contribution responds to this round's punishment, given own contribution."""
import pandas as pd, numpy as np
pd.set_option('display.width',200)
df = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
df = df.sort_values(['episode_id','participant_code','round_number'])
g = df.groupby(['episode_id','participant_code'])
df['c_next'] = g['contribution'].shift(-1)
df['ni_next'] = g['player_no_input'].shift(-1)
grp = df[df.player_no_input==0].groupby(['episode_id','round_number'])['contribution']
df['grp_sum'] = df.groupby(['episode_id','round_number'])['contribution'].transform('sum')
df['nvalid'] = df.groupby(['episode_id','round_number'])['player_no_input'].transform(lambda x: (1-x).sum())
df['others'] = (df.grp_sum - df.contribution)/(df.nvalid-1+ (df.player_no_input))
d = df[(df.player_no_input==0)&(df.ni_next==0)&df.c_next.notna()].copy()
d['dc'] = d.c_next - d.contribution
d['pbin'] = pd.cut(d.punishment, [-1,0,2,5,10,30])
d['cbin'] = pd.cut(d.contribution, [-1,4,9,14,19,20])
print(d.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
print(d.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='count'))
import numpy.linalg as la
X = np.column_stack([np.ones(len(d)), d.punishment, d.contribution, d.others, (d.punishment>0)])
b = la.lstsq(X, d.c_next, rcond=None)[0]; print('c_next ~ 1,p,c,others,p>0', b.round(3))
for ex in ['random_1','trail_rounds_2']:
    e=d[d.experiment_name==ex]; X = np.column_stack([np.ones(len(e)), e.punishment, e.contribution, e.others])
    print(ex, la.lstsq(X, e.c_next, rcond=None)[0].round(3))
# lagged effect: does punishment at t affect c at t+2 beyond c_{t+1}?
df['c_next2'] = g['contribution'].shift(-2)
d['c_next2']=df.loc[d.index,'c_next2']
e = d[d.c_next2.notna()]
X = np.column_stack([np.ones(len(e)), e.punishment, e.contribution, e.c_next, e.others])
print('c_t+2 ~ 1,p_t,c_t,c_t+1,others', la.lstsq(X, e.c_next2, rcond=None)[0].round(3))
