"""Replay v3's Sobol design on sur3_all (200 games per point); compare with v2's design on the same surrogate."""
import torch as th, numpy as np, sys, yaml
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7'); sys.path.insert(0,'src')
from aimanager.manager.rule import sobol_design
from sur3 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur3_all.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7_sweep.npz')['C'][:,:,0]).float()
G=200
def replay(path,keys,mkrule):
    pts=sobol_design(yaml.safe_load(open(path)),256); s=np.zeros(256)
    for i0 in range(0,256,64):
        par=th.tensor([[p[k] for k in keys] for p in pts[i0:i0+64]],dtype=th.float32).repeat_interleave(G,0)
        C0=C0all[th.randint(0,C0all.shape[0],(par.shape[0],),generator=th.Generator().manual_seed(i0))]
        s[i0:i0+64]=simulate(net,mkrule(par),C0,seed=i0).reshape(64,G).mean(1).numpy()
    print(path,'mean',s.mean().round(1),'pct 10/25/50/75/90/max',np.percentile(s,[10,25,50,75,90,100]).round(1))
    return pts,s
def v2(par):
    F,tg,gm,w=[par[:,i:i+1] for i in range(4)]
    return lambda C,P,t: F*((tg-C[...,-1]).clamp(min=0)/tg)**gm*(t<w).float()
def v3(par):
    F,gm,w,fl,E,sc=[par[:,i:i+1] for i in range(6)]
    def r(C,P,t):
        c=C[...,-1]; f=F*((20-c).clamp(min=0)/20)**gm
        return f*((t<w).float()+sc*((t>=w)&(t<E)).float()*(c<=fl).float())
    return r
replay('configs/managers/rule_based/pf-20261105-7-v2.yml',('free_rider_fine','target','curvature','window'),v2)
pts,s=replay('configs/managers/rule_based/pf-20261105-7-v3.yml',('free_rider_fine','curvature','window','net_floor','net_end','net_scale'),v3)
print('share net off',np.mean([p['net_end']<=p['window'] for p in pts]))
for i in np.argsort(-s)[:10]: print(round(s[i],1),{k:round(v,2) for k,v in pts[i].items()})
# crude per-parameter trend: top quartile mean vs all
keys=list(pts[0]); X=np.array([[p[k] for k in keys] for p in pts]); top=s>=np.percentile(s,75)
for j,k in enumerate(keys): print(k,'all',X[:,j].mean().round(2),'top quartile',X[top,j].mean().round(2))
