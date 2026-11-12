# Mean punishment per member the rule would deal on the zero-managed groups' contributions
# (reference sims, static: ignores how players would respond), for a grid of params.
import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d=d[d.run.str.endswith('zero_vs_ah')&(d.agent_group==0)&d.contribution_valid].copy()
d['gm']=d.groupby(['episode','round_number']).contribution.transform('mean')
print('share c<=4',(d.contribution<=4).mean().round(3))
for frac in [0.3,0.5,0.8,1.0]:
    for rate in [0.3,1,2,4]:
        for cap in [3,10,30]:
            p=np.floor(np.minimum(rate*np.clip(frac*d.gm-d.contribution,0,None),cap))
            print(frac,rate,cap,'mean p %.2f  share punished %.2f'%(p.mean(),(p>0).mean()))
