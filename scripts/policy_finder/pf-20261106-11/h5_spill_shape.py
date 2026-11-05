# v5: what feature of others' punishment drives the spillover - mean, max, or share punished? unpunished members, v4 sweep, rounds 0,1,4,5,8
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
d = load('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v4_sweep/', 'v4')
c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']; v=d['v'].astype(bool)
n=m.sum(1,keepdims=True)
others=((c*m).sum(1,keepdims=True)-c*m)/np.maximum(n-1,1)
pm=p*m
pothers=(pm.sum(1,keepdims=True)-pm)/np.maximum(n-1,1)
# others' max: since target is unpunished, max over group = max over others
pmax=pm.max(1,keepdims=True).repeat(8,1)
shp=((pm>0).sum(1,keepdims=True)-(pm>0))/np.maximum(n-1,1)
for t in [0,1,4,5,8]:
    s=m[:,:,t]&m[:,:,t+1]&v[:,:,t]&(p[:,:,t]==0)&(n[:,0,t]>2)[:,None]
    cols=[c[:,:,t],others[:,:,t],pothers[:,:,t],pmax[:,:,t],shp[:,:,t],n[:,0,t][:,None].repeat(8,1)]
    X=np.column_stack([np.ones(s.sum())]+[x[s] for x in cols])
    y=(c[:,:,t+1]-c[:,:,t])[s]
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    # bins of pmax among those with others' mean similar
    print(f't={t} n={s.sum()}: mean-p {b[3]:+.3f}  max-p {b[4]:+.3f}  share-punished {b[5]:+.2f}   (mean pmax {pmax[:,:,t][s].mean():.1f}, share {shp[:,:,t][s].mean():.2f})')
