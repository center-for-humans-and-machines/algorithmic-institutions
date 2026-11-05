"""Held-out check: sur3_all3 (trained on v1-v3) replays v4's final points (and v3's peak), common seeds."""
import numpy as np, torch as th, sys, json, time
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur3_all3.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v4_sweep.npz')['C'][:,:,0]).float()
G=int(sys.argv[1]) if len(sys.argv)>1 else 2000
def v4rule(F,gm,w,floor,E,nf,**k):
    def r(C,P,t):
        c=C[...,-1]
        if t<w: return F*((20-c).clamp(min=0)/20)**gm
        return nf*(c<=floor).float()*float(t<E)
    return r
def v3rule(F,gm,w,floor,E,sc):
    def r(C,P,t):
        c=C[...,-1]; f=F*((20-c).clamp(min=0)/20)**gm
        if t<w: return f
        return sc*f*(c<=floor).float()*float(t<E)
    return r
pts=json.load(open('plots/simulation/policy_finder/pf-20261105-7-v4_sweep/final/sweep.json'))['points']
C0=C0all[th.randint(0,C0all.shape[0],(G,),generator=th.Generator().manual_seed(3))]
t0=time.time()
for p in pts:
    q=p['params']
    pay=simulate(net,v4rule(q['free_rider_fine'],q['curvature'],q['window'],q['net_floor'],q['net_end'],q['net_fine']),C0,seed=7)
    print(p['name'][-6:],'real',round(p['pool'],1),'sur',round(pay.mean().item(),1),round(time.time()-t0),flush=True)
pay=simulate(net,v3rule(29.71,2.177,7,2,16,0.848),C0,seed=7); print('v3 peak real 656.0 sur',round(pay.mean().item(),1))
