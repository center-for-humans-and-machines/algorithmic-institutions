"""Does the surrogate reproduce 63's raw real-data table? Simulate v4-like ungated rules on sur3_all3 with
windows 4-10 (F 30, gm 2.2, net c<=2 at 22 to 16), then run 63's comparison on the simulated games."""
import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur3_all3.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v4_sweep.npz')['C'][:,:,0]).float()
def rule(w):
    def r(C,P,t):
        c=C[...,-1]
        if t<w: return 30*((20-c).clamp(min=0)/20)**2.2
        return 22*(c<=2).float()*float(t<16)
    return r
Cs=[];Ps=[]
for w in range(4,11):
    C0=C0all[th.randint(0,C0all.shape[0],(6000,),generator=th.Generator().manual_seed(w))]
    pay,C,P=simulate(net,rule(w),C0,seed=w,ret_hist=True); Cs.append(C.numpy());Ps.append(P.numpy())
C=np.concatenate(Cs);P=np.concatenate(Ps)
for lo,hi in [(10,13),(14,15)]:
  for t in [5,6,7,8]:
    gm=C[:,:,t].mean(1,keepdims=True); out=[f'c{lo:2d}-{hi} t{t}']
    for side,sm in [('above',C[:,:,t]>=gm+1),('below',C[:,:,t]<gm+1)]:
        base=(C[:,:,t]>=lo)&(C[:,:,t]<=hi)&sm; r={}
        for k,mm in [('f',base&(P[:,:,t]>=1)),('u',base&(P[:,:,t]==0))]:
            g,i=np.nonzero(mm)
            if len(g)<100: r=None;break
            own=(C[g,i,t+1:].mean(1)-C[g,i,t]).mean()
            pe=((C[g,:,t+1:].sum(1)-C[g,i,t+1:]).mean(1)-(C[g,:,t].sum(1)-C[g,i,t])).mean()
            r[k]=(own,pe,P[g,i,t].mean(),len(g))
        if r is None: out.append(f' | {side}: -'); continue
        out.append(f" | {side}: nf{r['f'][3]:5d} nu{r['u'][3]:5d} own {r['f'][0]-r['u'][0]:+4.1f} peers {r['f'][1]-r['u'][1]:+4.1f}")
    print(''.join(out))
