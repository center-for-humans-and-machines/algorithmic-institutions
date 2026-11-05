# v5: at block starts (t=4,8), newcomers vs stayers in my group: shortfall, natural drift when unpunished, response per point (narrow gap bins), leaving at the next switch; v4 sweep
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
d = load('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v4_sweep/', 'v4')
c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']; v=d['v'].astype(bool)
n=m.sum(1,keepdims=True)
others=((c*m).sum(1,keepdims=True)-c*m)/np.maximum(n-1,1); gap=c-others
for t in [4,8]:
    be=t+3; new=m[:,:,t]&~m[:,:,t-1]
    for lab,grp in [('new',new),('stay',m[:,:,t]&m[:,:,t-1])]:
        s=grp&v[:,:,t]&m[:,:,be]&(n[:,0,t]>2)[:,None]
        print(f't={t} {lab}: n={s.sum()} mean gap {gap[:,:,t][s].mean():+.2f} share below -2: {(gap[:,:,t][s]<-2).mean():.2f}')
        for lo,hi in [(-7,-4),(-4,-2)]:
            out=[]
            for plo,phi in [(0,1),(1,8),(8,16),(16,25),(25,31)]:
                ss=s&(gap[:,:,t]>=lo)&(gap[:,:,t]<hi)&(p[:,:,t]>=plo)&(p[:,:,t]<phi)
                if ss.sum()<200: out.append('    -    '); continue
                out.append(f'{(~m[:,:,be+1])[ss].mean():.3f}/{(c[:,:,be]-c[:,:,t])[ss].mean():+.1f}')
            print(f'   gap[{lo},{hi}) leave/dc-to-block-end by p 0,1-7,8-15,16-24,25-30:', ' '.join(out))
