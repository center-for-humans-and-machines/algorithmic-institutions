"""Defector-net variants on sur3_all (common seeds). fine(c)=F*((tg-c)+/tg)^gm in rounds < w;
after, only players with c <= floor, until round E (exclusive)."""
import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
tag=sys.argv[1] if len(sys.argv)>1 else 'all'
net=Net(); net.load_state_dict(th.load(f'scripts/policy_finder/pf-20261105-7/sur3_{tag}.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7_sweep.npz')['C'][:,:,0]).float()
G=3000
C0=C0all[th.randint(0,C0all.shape[0],(G,),generator=th.Generator().manual_seed(5))]
def mk(F,tg,gm,w,floor=-1,E=0,netF=None):
    nf=F if netF is None else netF
    def r(C,P,t):
        c=C[...,-1]; sh=((tg-c).clamp(min=0)/tg)
        f=F*sh**gm
        if t<w: return f
        if t<E: return (nf*sh**gm)*(c<=floor).float()
        return th.zeros_like(c)
    return r
def run(label,rule):
    pay,C,P=simulate(net,rule,C0,seed=11,ret_hist=True)
    print(f'{label:45s} {pay.mean():6.1f} +-{pay.std()/G**.5:.1f} c:',' '.join(f'{x:4.1f}' for x in C.mean((0,1))[::3].tolist()),' p/pl:',round(P.mean((0,1)).sum().item(),1),flush=True)
    return pay.mean().item()
run('zero',lambda C,P,t: th.zeros_like(C[...,-1]))
run('v1 gp',lambda C,P,t: 2.08*(14-C[...,-1]).clamp(min=0)*float(t<9)*(1-0.18*t/9))
run('v2 s132',mk(31.26,19,2.15,11))
run('v2 s078',mk(26.5,20,2.04,9))
for w in [6,9]:
    for floor in [2,4,6,8]:
        for E in [14,17,20,23]:
            run(f's132 w{w} floor{floor} E{E}',mk(31.26,19,2.15,w,floor,E))
