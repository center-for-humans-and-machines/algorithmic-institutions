"""Simulate v1's 256 design points + zero on surrogate v2, compare with the real sweep scores."""
import numpy as np, torch as th, json, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur2 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur2.pt'))
a=np.load('/tmp/pf7_arr.npz'); C0all=th.tensor(a['C'][:,:,0])
sw=json.load(open('plots/simulation/policy_finder/pf-20261105-7_sweep/sweep.json'))
pts=sw['points']; G=300
par=th.tensor([[p['params'][k] for k in ('target','slope','window','taper')] for p in pts],dtype=th.float32)
par=par.repeat_interleave(G,0)
tg,sl,wi,ta=[par[:,i:i+1] for i in range(4)]
def rule(C,P,t):
    sh=(tg-C[...,-1]).clamp(min=0)
    return sl*sh*(t<wi).float()*(1-ta*t/wi).clamp(min=0)
g=th.Generator().manual_seed(1)
C0=C0all[th.randint(0,C0all.shape[0],(par.shape[0],),generator=g)]
pay,C,P=simulate(net,rule,C0)
pay=pay.reshape(len(pts),G).mean(1)*24/24
s=pay.numpy()*1.0
# pay is per-game sum? simulate sums round means -> total
real=np.array([p['pool'] for p in pts])
print('corr',np.corrcoef(s,real)[0,1], 'mean sur',s.mean(),'mean real',real.mean())
o=np.argsort(-real)
for i in o[:10]: print(round(real[i],1),round(s[i],1),pts[i]['params'])
print('sur top:')
for i in np.argsort(-s)[:10]: print(round(real[i],1),round(s[i],1),pts[i]['params'])
zero=lambda C,P,t: th.zeros_like(C[...,-1])
pz,Cz,_=simulate(net,zero,C0all[:3000])
print('zero',pz.mean().item(), 'c by round',Cz.mean((0,1)).numpy().round(1))
