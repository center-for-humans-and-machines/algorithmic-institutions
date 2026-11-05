"""How the humanlike sim manager and the pilot managers punish, by contribution."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
s=pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
s=s[s.contribution_valid]
h=s[s.run.str.contains('humanlike')]
h['cb']=pd.cut(h.contribution,[-1,0,4,9,14,19,20])
print('humanlike mean punishment by contribution bin'); print(h.groupby('cb').punishment.agg(['mean','size',lambda x:(x>0).mean()]))
print('by round (contrib, pun)'); print(h.groupby('round_number')[['contribution','punishment']].mean().T.round(1))
for run,g in s.groupby('run'):
    g=g.copy(); g['cb']=pd.cut(g.contribution,[-1,0,4,9,14,19,20])
    print(run,'contribution distribution by round group')
    g['rb']=g.round_number//6
    print(g.groupby('rb').cb.value_counts(normalize=True).unstack().round(2))
p=pd.read_csv('experiments/pilot_random1_player_round_slim.csv'); p=p[p.player_no_input==0]
p['cb']=pd.cut(p.contribution,[-1,0,4,9,14,19,20])
for n,g in p.groupby('experiment_name'):
    print('PILOT',n); print(g.groupby('cb').punishment.agg(['mean','size']))
    print(g.groupby('round_number')[['contribution','punishment','payoff']].mean().T.round(1))
