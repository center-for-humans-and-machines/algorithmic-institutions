# v5: exit cost of my rounds 0-1 punishment, by opponent (who punish 0 / 3.8 / 10.8 per head in round 0): leave at first switch vs own p in round 0 and 1, and vs others' p; also the round-1 own response; v3+v4 sweeps
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
for tag in ['v3','v4']:
    d = load(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-{tag}_sweep/', tag)
    c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']; v=d['v'].astype(bool); opp=d['opp']
    n=m.sum(1,keepdims=True)
    others=((c*m).sum(1,keepdims=True)-c*m)/np.maximum(n-1,1)
    pm=p*m; po=(pm.sum(1,keepdims=True)-pm)/np.maximum(n-1,1)
    for o in [1,2,3,4]:
        s=(opp==o)[:,None]&m[:,:,0]&m[:,:,3]&v[:,:,0]
        f=lambda a:a[s]
        g0=f(c[:,:,0]-others[:,:,0]); g1=f(c[:,:,1]-others[:,:,1])
        X=np.column_stack([np.ones(s.sum()),np.minimum(g0,0),np.maximum(g0,0),np.minimum(g1,0),np.maximum(g1,0),f(c[:,:,1]),f(p[:,:,0]),f(p[:,:,1]),f(po[:,:,0]),f(po[:,:,1])])
        lv=f(~m[:,:,4]).astype(float)
        b=np.linalg.lstsq(X,lv,rcond=None)[0]
        # inflow: members at round 4 from the other group
        inflow=((~m[:,:,3])&m[:,:,4]).sum(1)[opp==o].mean(); outflow=(m[:,:,3]&~m[:,:,4]).sum(1)[opp==o].mean()
        print(f'{tag} opp{o}: base leave {lv.mean():.3f} | own p0 {b[6]:+.4f} own p1 {b[7]:+.4f} | others p0 {b[8]:+.4f} others p1 {b[9]:+.4f} | in {inflow:.2f} out {outflow:.2f}')
