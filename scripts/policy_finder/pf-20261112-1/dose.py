# Mean punishment per member-round the rule would deal on zero-managed groups' contributions (rounds >= t_start, < 23)
import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet',columns=['run','contribution','round_number','contribution_valid'])
d=d[d.run.str.endswith('zero_vs_zero')&d.contribution_valid&(d.round_number>=4)&(d.round_number<23)]
c=d.contribution.values
for thr,k,cap in [(5,0.5,3),(8,0.5,5),(10,1,5),(10,0.3,3),(14,1,10),(16,3,15)]:
    p=np.floor(np.minimum(cap,k*np.clip(thr-c,0,None)))
    print(thr,k,cap,'mean %.2f share>0 %.2f'%(p.mean(),(p>0).mean()))
