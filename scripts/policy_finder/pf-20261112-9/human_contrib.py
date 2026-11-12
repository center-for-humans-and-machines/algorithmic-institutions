import pandas as pd, numpy as np
d=pd.read_csv('experiments/2group_8agent_50ep.csv')
d=d.drop_duplicates(['session','episode','round_number','participant_code'])
d['game']=d.session.astype(str)+'_'+d.episode.astype(str)
d=d.sort_values(['game','player_id','round_number'])
d['nc']=d.groupby(['game','player_id']).contribution.shift(-1)
d['ng']=d.groupby(['game','player_id']).group_id.shift(-1)
x=d[(d.ng==d.group_id)&(d.player_no_input==0)].dropna(subset=['nc']).copy()
x['dc']=x.nc-x.contribution
x['pb']=pd.cut(x.punishment,[-.1,0,2,5,10,31])
x['cb']=pd.cut(x.contribution,[-.1,4,9,14,19,20])
print(x.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean',observed=False).round(2))
print(x.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size',observed=False))
# what managers did: punishment by contribution level
print(d.groupby(pd.cut(d.contribution,[-.1,0,4,9,14,19,20]),observed=False).punishment.agg(['mean','size']).round(2))
# group-level: contribution relative to group mean
d['gm']=d.groupby(['game','round_number','group_id']).contribution.transform('mean')
d['rel']=d.contribution-d.gm
print(d.groupby(pd.cut(d.rel,[-21,-8,-4,-1,1,4,21]),observed=False).punishment.agg(['mean','size']).round(2))
print('mean c by round', d.groupby('round_number').contribution.mean().round(1).tolist())
print('mean p by round', d.groupby('round_number').punishment.mean().round(1).tolist())
