# v5: dose shape within narrow shortfall bins (dose varies via rate/delta across design points): leaving at the block's switch and change to t+1 / block end, v3+v4 sweeps, t=0,4
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
for t in [0,1,4]:
  for tag in ['v3','v4']:
    d = load(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-{tag}_sweep/', tag)
    c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']; v=d['v'].astype(bool)
    n=m.sum(1,keepdims=True)
    others=((c*m).sum(1,keepdims=True)-c*m)/np.maximum(n-1,1)
    gap=c-others; be=(t//4)*4+3
    for lo,hi in [(-7,-5),(-4,-2.5)]:
        s=m[:,:,t]&v[:,:,t]&m[:,:,be]&(n[:,0,t]>2)[:,None]&(gap[:,:,t]>=lo)&(gap[:,:,t]<hi)
        out=[]
        for plo,phi in [(0,1),(1,6),(6,12),(12,18),(18,24),(24,29),(29,31)]:
            ss=s&(p[:,:,t]>=plo)&(p[:,:,t]<phi)
            if ss.sum()<300: out.append('     -      '); continue
            lv=(~m[:,:,be+1])[ss].mean(); s1=ss&m[:,:,t+1]
            dc=(c[:,:,t+1]-c[:,:,t])[s1].mean(); dce=(c[:,:,be]-c[:,:,t])[ss].mean()
            out.append(f'{lv:.3f}/{dc:+.1f}/{dce:+.1f}')
        print(f'{tag} t={t} gap[{lo},{hi}) leave/dc1/dcEnd by p 0,1-5,6-11,12-17,18-23,24-28,30:', ' '.join(out))
