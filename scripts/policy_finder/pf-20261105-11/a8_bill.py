# Round-0 and zero-sim-state punishment bill of the drafted rule family p = P*(max(0,T-c)/T)^g
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
z=d[d.run.str.contains('zero')&d.contribution_valid]
c0=z[z.round_number==0].contribution.values
print('round0 c quantiles', np.quantile(c0,[.1,.25,.5,.75,.9]), 'share c<10', (c0<10).mean(), 'share c=20', (c0==20).mean())
for T in (10,14,17,20):
  for P in (3,8,15):
    for g in (0.3,1.0):
      dd=np.clip(T-c0,0,None)/T
      p=np.floor(np.clip(P*dd**g*(dd>0),0,30))
      print(f'T={T} P={P} g={g}: mean p round0 {p.mean():.2f}  cost/round(sum-pay per player) {2*p.mean():.2f}  share punished {(p>0).mean():.2f}')
