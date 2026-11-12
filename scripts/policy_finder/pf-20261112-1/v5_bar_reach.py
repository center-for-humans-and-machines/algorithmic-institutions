# v5: in zero-managed sweep games (v3 unpunished points), at decisions 7/11/15 with my group >= 7 members, the
# lowest valid member's window mean: how many games a bar of b would touch (first push), b = 0..14.
import sys, numpy as np
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261112-1')
from v3_load import load
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
d=load(B+'pf-20261112-1-v3_sweep/per_round_p1of2.parquet')
own=(d['G']==d['mine'][:,None,None]); pun=(d['P']*own).sum((1,2))
I=pun==0; C=d['C'][I].astype(float); V=d['V'][I]; O=own[I]
print('games',I.sum())
first=np.full(I.sum(),99.)
lows={}
for t in [7,11,15]:
    w=slice(t-3,t+1); vv=V[:,:,w]; cm=(C[:,:,w]*vv).sum(-1)/np.clip(vv.sum(-1),1,None)
    elig=O[:,:,t]&(V[:,:,t]>0); size=O[:,:,t].sum(1)
    low=np.where(elig,cm,99).min(1); low[size<7]=99
    lows[t]=low
for b in range(0,15):
    touched=np.zeros(len(first),bool)
    for t in [7,11,15]: touched|=lows[t]<=b
    print(b, 'touched share %.3f'%touched.mean(), ' at7 %.3f at11 %.3f at15 %.3f'%tuple((lows[t]<=b).mean() for t in [7,11,15]))
