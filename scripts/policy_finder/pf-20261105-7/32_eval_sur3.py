"""Replay v1's and v2's real design points on a sur3 model; correlate with the real sweep scores.
argv[1]: model tag (v1/all). argv[2]: games per point."""
import numpy as np, torch as th, json, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
tag=sys.argv[1]; G=int(sys.argv[2])
net=Net(); net.load_state_dict(th.load(f'scripts/policy_finder/pf-20261105-7/sur3_{tag}.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7_sweep.npz')['C'][:,:,0]).float()
def v1rule(par):
    tg,sl,wi,ta=[par[:,i:i+1] for i in range(4)]
    return lambda C,P,t: sl*(tg-C[...,-1]).clamp(min=0)*(t<wi).float()*(1-ta*t/wi).clamp(min=0)
def v2rule(par):
    F,tg,gm,w=[par[:,i:i+1] for i in range(4)]
    return lambda C,P,t: F*((tg-C[...,-1]).clamp(min=0)/tg)**gm*(t<w).float()
out={}
for v,keys,mk in [('pf-20261105-7',('target','slope','window','taper'),v1rule),('pf-20261105-7-v2',('free_rider_fine','target','curvature','window'),v2rule)]:
    pts=json.load(open(f'plots/simulation/policy_finder/{v}_sweep/sweep.json'))['points']
    real=np.array([p['pool'] for p in pts]); s=np.zeros(len(pts))
    for i0 in range(0,len(pts),64):
        sub=pts[i0:i0+64]
        par=th.tensor([[p['params'][k] for k in keys] for p in sub],dtype=th.float32).repeat_interleave(G,0)
        C0=C0all[th.randint(0,C0all.shape[0],(par.shape[0],),generator=th.Generator().manual_seed(i0))]
        pay=simulate(net,mk(par),C0,seed=i0)
        s[i0:i0+len(sub)]=pay.reshape(len(sub),G).mean(1).numpy()
    print(v,'corr',np.corrcoef(s,real)[0,1].round(3),'mean sur',s.mean().round(1),'real',real.mean().round(1))
    o=np.argsort(-real)[:8]
    print('  real top:',[(round(real[i],1),round(s[i],1)) for i in o])
    o=np.argsort(-s)[:8]
    print('  sur top:',[(round(real[i],1),round(s[i],1),{k:round(x,2) for k,x in pts[i]['params'].items()}) for i in o])
    out[v]=(s.tolist(),real.tolist())
pz=simulate(net,lambda C,P,t: th.zeros_like(C[...,-1]),C0all[:3000],seed=9)
print('zero',pz.mean().item())
json.dump(out,open(f'/tmp/pf7/eval_{tag}.json','w'))
