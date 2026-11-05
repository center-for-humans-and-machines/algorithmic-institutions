import pandas as pd, numpy as np
d = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
print(d.shape); print(d.columns.tolist())
print(d.groupby('experiment_name').agg(groups=('global_group_id','nunique'), eps=('episode_id','nunique'), rows=('payoff','size')))
v = d[d.player_no_input==0]
print('timeout rate', d.player_no_input.mean(), d.groupby('experiment_name').player_no_input.mean().to_dict())
print(v.groupby(['experiment_name','round_number']).agg(c=('contribution','mean'),p=('punishment','mean'),pay=('payoff','mean'),n=('payoff','size')).unstack(0).round(2))
print(v.groupby('experiment_name')[['contribution','punishment','payoff']].mean())
print('punishment distribution by contribution bin (human mgr)')
h = v[v.experiment_name=='trail_rounds_2']
print(h.groupby(pd.cut(h.contribution,[-1,0,5,10,15,19,20])).punishment.agg(['mean','size', lambda x:(x>0).mean()]))
r = v[v.experiment_name=='random_1']
print(r.groupby(pd.cut(r.contribution,[-1,0,5,10,15,19,20])).punishment.agg(['mean','size', lambda x:(x>0).mean()]))
