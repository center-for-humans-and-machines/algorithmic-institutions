"""Next-round contribution change vs punishment received, in the humanlike sim and in pilots."""
import pandas as pd, numpy as np
def prep(d, pcol='participant_code', ep='episode', valid='contribution_valid'):
    d=d.sort_values([ep,pcol,'round_number']).copy()
    g=d.groupby([ep,pcol])
    d['c_next']=g.contribution.shift(-1); d['v_next']=g[valid].shift(-1)
    d['others']=(d.groupby([ep,'round_number']).contribution.transform('sum')-d.contribution)/3
    d=d[d[valid].astype(bool) & d.v_next.astype(bool)]
    d['dc']=d.c_next-d.contribution
    d['pb']=pd.cut(d.punishment,[-1,0,2,4,6,9,14,30])
    d['cb']=pd.cut(d.contribution,[-1,4,9,14,19,20])
    return d
s=pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
s=s[s.run.str.contains('humanlike')]
s=prep(s)
print('SIM humanlike: mean dc by contribution bin x punishment bin')
print(s.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean',observed=True).round(2))
print(s.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size',observed=True))
h=pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
h['valid']=1-h.player_no_input
for name,g in h.groupby('experiment_name'):
    g=prep(g,ep='episode_id',valid='valid')
    print('PILOT',name)
    print(g.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean',observed=True).round(2))
    print(g.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size',observed=True))
