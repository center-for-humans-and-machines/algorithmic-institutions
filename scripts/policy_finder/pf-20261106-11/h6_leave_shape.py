# v5: is leaving after own punishment linear in the dose, or does being punished at all carry a fixed cost? v4 sweep, punishing rounds 0,1,4,5,8; leave at the block's switch; also own next-round response with the same split
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
d = load('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v4_sweep/', 'v4')
c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']; v=d['v'].astype(bool)
n=m.sum(1,keepdims=True)
others=((c*m).sum(1,keepdims=True)-c*m)/np.maximum(n-1,1)
pm=p*m; pothers=(pm.sum(1,keepdims=True)-pm)/np.maximum(n-1,1)
for t in [0,1,4,5,8]:
    be=(t//4)*4+3
    s=m[:,:,t]&m[:,:,be]&v[:,:,t]&(n[:,0,t]>1)[:,None]
    g=(c-others)[:,:,t][s]; pp=p[:,:,t][s]
    X=np.column_stack([np.ones(s.sum()),np.minimum(g,0),np.maximum(g,0),c[:,:,t][s],pothers[:,:,t][s],(pp>0).astype(float),pp,np.minimum(pp,10)])
    lv=(~m[:,:,be+1])[s].astype(float)
    b=np.linalg.lstsq(X,lv,rcond=None)[0]
    s1=s&m[:,:,t+1]
    g1=(c-others)[:,:,t][s1]; p1=p[:,:,t][s1]
    X1=np.column_stack([np.ones(s1.sum()),np.minimum(g1,0),np.maximum(g1,0),c[:,:,t][s1],pothers[:,:,t][s1],(p1>0).astype(float),p1,np.minimum(p1,10)])
    b1=np.linalg.lstsq(X1,(c[:,:,t+1]-c[:,:,t])[s1],rcond=None)[0]
    print(f't={t} leave base {lv.mean():.3f}: 1[p>0] {b[5]:+.3f}  per pt {b[6]:+.4f}  extra per pt below 10 {b[7]:+.4f} | dc: 1[p>0] {b1[5]:+.2f} per pt {b1[6]:+.3f} extra<10 {b1[7]:+.3f}')
