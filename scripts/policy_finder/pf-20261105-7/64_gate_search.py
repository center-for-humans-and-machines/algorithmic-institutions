"""Random search of the gated family on sur3_all3 (2 seed sets x 2000 games per point): F, gm, w, m, floor, E, nf.
Writes /tmp/pf7/gate_search.npy rows [F,gm,w,m,floor,E,nf,score]."""
import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur3_all3.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v4_sweep.npz')['C'][:,:,0]).float()
G=2000
def rule(F,gm,w,m,floor,E,nf):
    def r(C,P,t):
        c=C[...,-1]
        if t<w: return F*((20-c).clamp(min=0)/20)**gm*(c<c.mean(1,keepdim=True)+m).float()
        return nf*(c<=floor).float()*float(t<E)
    return r
C0s=[C0all[th.randint(0,C0all.shape[0],(G,),generator=th.Generator().manual_seed(s))] for s in (21,22)]
rng=np.random.default_rng(0); rows=[]
N=int(sys.argv[1])
for k in range(N):
    q=[rng.uniform(25,50),np.exp(rng.uniform(np.log(1.4),np.log(3.0))),rng.integers(4,12),rng.uniform(-2,8),rng.integers(0,5),rng.integers(12,21),rng.uniform(12,30)]
    sc=np.mean([simulate(net,rule(*q),C0s[j],seed=31+j).mean().item() for j in range(2)])
    rows.append(q+[sc])
    if k%20==0: print(k,flush=True); np.save('/tmp/pf7/gate_search.npy',np.array(rows))
np.save('/tmp/pf7/gate_search.npy',np.array(rows))
