# v4: punishment in rounds 2-3 (block 0's tail) in the v1 sweep (which punished round 2, and round 3 at lam): response and exit cost per point, versus rounds 0-1
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
tag=sys.argv[1] if len(sys.argv)>1 else 'v1'
D={'v1':'pf-20261106-11_sweep','v2':'pf-20261106-11-v2_sweep'}[tag]
d = load(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/{D}/', tag)
sub=np.arange(0,len(d['c']),2)
c=d['c'][sub].astype(np.float32); p=d['p'][sub].astype(np.float32); m=d['mine'][sub]
n=m.sum(1,keepdims=True)
others=((c*m).sum(1,keepdims=True)-c*m)/np.maximum(n-1,1)
gap=c-others
for t in [0,1,2,3,4,5,6]:
    be=(t//4)*4+3
    s=m[:,:,t]&m[:,:,be]&(n[:,0,t]>1)[:,None]
    lv=(~m[:,:,be+1])[s].astype(float)
    X=np.c_[np.ones(s.sum()),np.minimum(gap[:,:,t],0)[s],np.maximum(gap[:,:,t],0)[s],p[:,:,t][s],c[:,:,t][s]]
    bl=np.linalg.lstsq(X,lv,rcond=None)[0]
    s1=s&m[:,:,t+1]
    X1=np.c_[np.ones(s1.sum()),np.minimum(gap[:,:,t],0)[s1],np.maximum(gap[:,:,t],0)[s1],p[:,:,t][s1],c[:,:,t][s1]]
    b=np.linalg.lstsq(X1,(c[:,:,t+1]-c[:,:,t])[s1],rcond=None)[0]
    print(f'{tag} t={t}: mean p {p[:,:,t][s].mean():.2f}  dc next per p {b[3]:.3f}  leave at switch per p {bl[3]:.4f} (base {lv.mean():.3f})')
