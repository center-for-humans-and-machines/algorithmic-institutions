# v4: why does punishment's effect fade by round 4? split by whether the player was punished before (any group), and check other managers' groups too
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
d = load('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v3_sweep/', 'v3')
sub=np.arange(0,len(d['c']),2)   # half the games, memory
c=d['c'][sub].astype(np.float32); p=d['p'][sub].astype(np.float32); M=d['mine'][sub]; opp=d['opp'][sub]
cum=np.concatenate([np.zeros_like(p[:,:,:1]),np.cumsum(p,2)[:,:,:-1]],2)  # punishment before round t
for side in ['mine','other']:
    m = M if side=='mine' else ~M
    n=m.sum(1,keepdims=True)
    others=((c*m).sum(1,keepdims=True)-c*m)/np.maximum(n-1,1)
    gap=c-others
    for t in [0,1,4,5,8,12]:
        for prior in ['none','some']:
            s=m[:,:,t]&m[:,:,t+1]&(n[:,0,t]>1)[:,None]&(gap[:,:,t]<-1)
            s&= (cum[:,:,t]==0) if prior=='none' else (cum[:,:,t]>0)
            if side=='other': s&=(opp!=4)[:,None]
            if s.sum()<2000: continue
            X=np.c_[np.ones(s.sum()),gap[:,:,t][s],p[:,:,t][s],c[:,:,t][s]]
            b=np.linalg.lstsq(X,(c[:,:,t+1]-c[:,:,t])[s],rcond=None)[0]
            print(f'{side:5s} t={t:2d} prior={prior}: n={s.sum():7d} mean p={p[:,:,t][s].mean():5.2f} gap {gap[:,:,t][s].mean():5.2f} dc/p={b[2]:.3f}')
