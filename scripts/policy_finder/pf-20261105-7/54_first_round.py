"""First-round boost on sur3_all3 (common seeds, two seed sets): round-0 fine multiplier r0, and a
boost over the first k rounds; with the opening schedule of v3's peak and a constant net of 25."""
import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur3_all3.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v3_sweep.npz')['C'][:,:,0]).float()
G=3000
def rule(F=29.7,gm=2.18,w=7,floor=2,E=16,nf=25.,r0=1.0,k=1):
    def r(C,P,t):
        c=C[...,-1]; f=F*((20-c).clamp(min=0)/20)**gm
        if t<k: return f*r0
        if t<w: return f
        return nf*(c<=floor).float()*float(t<E)
    return r
for s in [5,6]:
    C0=C0all[th.randint(0,C0all.shape[0],(G,),generator=th.Generator().manual_seed(s))]
    def run(label,r):
        pay,C,P=simulate(net,r,C0,seed=11+s,ret_hist=True)
        print(f'seed{s} {label:28s} {pay.mean():6.1f} c:',' '.join(f'{x:4.1f}' for x in C.mean((0,1))[:8].tolist()),' p:',' '.join(f'{x:4.1f}' for x in P.mean((0,1))[:8].tolist()),' p/pl:',round(P.mean((0,1)).sum().item(),1),flush=True)
    run('base',rule())
    for r0 in [1.4,2.0,3.0]: run(f'r0 x{r0}',rule(r0=r0))
    for r0,k in [(1.5,2),(2.0,2)]: run(f'first {k} x{r0}',rule(r0=r0,k=k))
    run('r0 x2 gm1.8 F25',rule(F=25,gm=1.8,r0=2.0))
