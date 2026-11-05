"""Replay v2's actual Sobol design on surrogate v2 (all points batched)."""
import torch as th, numpy as np, sys, yaml
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7'); sys.path.insert(0,'src')
from aimanager.manager.rule import sobol_design
from sur2 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur2.pt',weights_only=True))
rule=yaml.safe_load(open('configs/managers/rule_based/pf-20261105-7-v2.yml'))
pts=sobol_design(rule,256); G=400
par=th.tensor([[p[k] for k in ('free_rider_fine','target','curvature','window')] for p in pts],dtype=th.float32).repeat_interleave(G,0)
F,tg,gm,w=[par[:,i:i+1] for i in range(4)]
def r(C,P,t): return F*((tg-C[...,-1]).clamp(min=0)/tg)**gm*(t<w).float()
C0all=th.tensor(np.load('/tmp/pf7_arr.npz')['C'][:,:,0])
C0=C0all[th.randint(0,C0all.shape[0],(par.shape[0],),generator=th.Generator().manual_seed(2))]
pay,_,_=simulate(net,r,C0,seed=2)
s=pay.reshape(len(pts),G).mean(1).numpy()
print('mean',s.mean().round(1),'quantiles',np.percentile(s,[10,25,50,75,90,100]).round(1), 'share>640',(s>640).mean().round(2))
