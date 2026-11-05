"""More structural candidates on sur3_all3 (common seeds, two seed sets), base as in 60.
lowest: in the opening only the group's lowest contributor(s) get the schedule (others 0);
below_mean: opening schedule only for players below the group mean; net_taper: net fine x (rounds left in net / net length);
net_scaled: net fine grows with depth (c=0 full, c=floor 0.5x)."""
import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur3_all3.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v4_sweep.npz')['C'][:,:,0]).float()
G=3000
def rule(F=30,gm=2.2,w=7,floor=2,E=16,nf=22.,mode='base',x=1.0):
    def r(C,P,t):
        c=C[...,-1]
        sched=F*((20-c).clamp(min=0)/20)**gm
        if t<w:
            if mode=='lowest' and t>=x: return sched*(c<=c.min(1,keepdim=True).values).float()
            if mode=='below_mean' and t>=x: return sched*(c<c.mean(1,keepdim=True)).float()
            return sched
        netf=nf*(c<=floor).float()*float(t<E)
        if mode=='net_taper': netf=netf*(1-x*(t-w)/(E-w))
        return netf
    return r
res={}
for s in [5,6]:
    C0=C0all[th.randint(0,C0all.shape[0],(G,),generator=th.Generator().manual_seed(s))]
    def run(label,r):
        pay,C,P=simulate(net,r,C0,seed=11+s,ret_hist=True)
        res.setdefault(label,[]).append(pay.mean().item())
    run('base',rule())
    for x in [0,1,2,3]: run(f'lowest from round {x}',rule(mode='lowest',x=x))
    for x in [0,1,2,3]: run(f'below_mean from round {x}',rule(mode='below_mean',x=x))
    for x in [0.5,1.0]: run(f'net_taper {x}',rule(mode='net_taper',x=x))
    for x in [0.5,1.0]: run(f'net_taper {x} nf30',rule(mode='net_taper',x=x,nf=30))
    for E in [14,18,20]: run(f'end{E}',rule(E=E))
for l,v in res.items(): print(f'{l:34s} {np.mean(v):6.1f} {np.mean(v)-np.mean(res["base"]):+5.1f}  ({v[0]-res["base"][0]:+.1f},{v[1]-res["base"][1]:+.1f})')
