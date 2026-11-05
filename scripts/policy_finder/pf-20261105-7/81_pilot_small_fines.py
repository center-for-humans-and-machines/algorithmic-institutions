# Pilots (the clones' training data): next-round change of players by own c band and fine size.
import pandas as pd, numpy as np
d=pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
d=d.sort_values(['global_group_id','episode','participant_code','round_number'])
g=d.groupby(['global_group_id','episode','participant_code'])
d['c_next']=g.contribution.shift(-1); d['valid_next']=g.player_no_input.shift(-1)==0
d['c_rest']=g.contribution.transform(lambda s: s[::-1].expanding().mean()[::-1].shift(-1))
x=d[(d.player_no_input==0)&d.valid_next&(d.manager_no_input!=1)].copy()
x['band']=pd.cut(x.contribution,[-1,4,9,14,17,19,20],labels=['0-4','5-9','10-14','15-17','18-19','20'])
x['fb']=pd.cut(x.punishment,[-1,0,2,5,10,30],labels=['0','1-2','3-5','6-10','11+'])
x['d1']=x.c_next-x.contribution; x['dr']=x.c_rest-x.contribution
t=x.groupby(['experiment_name','band','fb'],observed=True).agg(n=('d1','size'),d1=('d1','mean'),drest=('dr','mean')).round(2)
print(t.to_string())
