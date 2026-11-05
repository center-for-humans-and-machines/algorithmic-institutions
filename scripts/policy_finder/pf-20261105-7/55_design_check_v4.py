"""Replay v3's and v4's Sobol designs on sur3_all3 (200 games per point, same seeds per slot)."""
import torch as th, numpy as np, sys, yaml
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7'); sys.path.insert(0,'src')
from aimanager.manager.rule import sobol_design
from sur3 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur3_all3.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v3_sweep.npz')['C'][:,:,0]).float()
G=200
def replay(path,keys,mk):
    pts=sobol_design(yaml.safe_load(open(path)),256); s=np.zeros(256)
    for i0 in range(0,256,64):
        par=th.tensor([[p[k] for k in keys] for p in pts[i0:i0+64]],dtype=th.float32).repeat_interleave(G,0)
        C0=C0all[th.randint(0,C0all.shape[0],(par.shape[0],),generator=th.Generator().manual_seed(i0))]
        s[i0:i0+64]=simulate(net,mk(par),C0,seed=i0).reshape(64,G).mean(1).numpy()
    print(path,'mean',s.mean().round(1),'pct 10/25/50/75/90/max',np.percentile(s,[10,25,50,75,90,100]).round(1))
    return pts,s
def v3(par):
    F,gm,w,fl,E,sc=[par[:,i:i+1] for i in range(6)]
    def r(C,P,t):
        c=C[...,-1]; f=F*((20-c).clamp(min=0)/20)**gm
        return f*((t<w).float()+sc*((t>=w)&(t<E)).float()*(c<=fl).float())
    return r
def v4(par):
    F,gm,w,fl,E,nf=[par[:,i:i+1] for i in range(6)]
    def r(C,P,t):
        c=C[...,-1]; f=F*((20-c).clamp(min=0)/20)**gm
        return f*(t<w).float()+nf*((t>=w)&(t<E)).float()*(c<=fl).float()
    return r
replay('configs/managers/rule_based/pf-20261105-7-v3.yml',('free_rider_fine','curvature','window','net_floor','net_end','net_scale'),v3)
pts,s=replay('configs/managers/rule_based/pf-20261105-7-v4.yml',('free_rider_fine','curvature','window','net_floor','net_end','net_fine'),v4)
for i in np.argsort(-s)[:8]: print(round(s[i],1),{k:round(v,2) for k,v in pts[i].items()})
