# v5: spillover - does punishment dealt to OTHER members change an unpunished member's next contribution and leaving? v3+v4 sweeps, rounds 0,1,4
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
for tag in ['v3','v4']:
    d = load(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-{tag}_sweep/', tag)
    c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']; v=d['v'].astype(bool)
    n=m.sum(1,keepdims=True)
    others=((c*m).sum(1,keepdims=True)-c*m)/np.maximum(n-1,1)
    pothers=((p*m).sum(1,keepdims=True)-p*m)/np.maximum(n-1,1)
    for t in [0,1,4]:
        be=(t//4)*4+3
        s=m[:,:,t]&m[:,:,t+1]&v[:,:,t]&(p[:,:,t]==0)&(n[:,0,t]>1)[:,None]
        X=np.c_[np.ones(s.sum()),c[:,:,t][s],others[:,:,t][s],pothers[:,:,t][s],n[:,0,t][:,None].repeat(8,1)[s]]
        b=np.linalg.lstsq(X,(c[:,:,t+1]-c[:,:,t])[s],rcond=None)[0]
        s2=m[:,:,t]&m[:,:,be]&v[:,:,t]&(p[:,:,t]==0)&(n[:,0,t]>1)[:,None]
        X2=np.c_[np.ones(s2.sum()),c[:,:,t][s2],others[:,:,t][s2],pothers[:,:,t][s2],n[:,0,t][:,None].repeat(8,1)[s2]]
        b2=np.linalg.lstsq(X2,(~m[:,:,be+1])[s2].astype(float),rcond=None)[0]
        b3=np.linalg.lstsq(X2,(c[:,:,be]-c[:,:,t])[s2],rcond=None)[0]
        print(f'{tag} t={t} n={s.sum()} mean others-p {pothers[:,:,t][s].mean():.2f}: dc_next per pt others-p {b[3]:+.3f}, dc to block end {b3[3]:+.3f}, leave {b2[3]:+.4f} (own c {b[1]:+.2f}, others mean {b[2]:+.2f})')
