"""More structural candidates on sur3_all3 (common seeds): round-0 fine multiplier, a drop net
(fine a player falling D below own best since the opening), a group trigger (re-open the schedule
while the group's mean is below a level), and a refined v3 grid for the narrowed ranges."""
import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur3_all3.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v3_sweep.npz')['C'][:,:,0]).float()
G=3000
C0=C0all[th.randint(0,C0all.shape[0],(G,),generator=th.Generator().manual_seed(5))]
def rule(F=29.7,gm=2.18,w=7,floor=2,E=16,nf=25.,r0=1.0,drop=None,dropF=0.,dropE=16,trig=None,trigE=16):
    def r(C,P,t):
        c=C[...,-1]; f=F*((20-c).clamp(min=0)/20)**gm
        if t==0: return f*r0
        if t<w: return f
        out=th.zeros_like(c)
        if t<E: out=out+nf*(c<=floor).float()
        if drop is not None and t<dropE:
            best=C[...,:-1].max(-1).values
            out=th.maximum(out,dropF*((best-c)>=drop).float()*(c>floor).float())
        if trig is not None and t<trigE:
            g=c.mean(-1,keepdim=True)
            out=th.maximum(out,f*(g<trig).float())
        return out
    return r
def run(label,r):
    pay,C,P=simulate(net,r,C0,seed=11,ret_hist=True)
    print(f'{label:40s} {pay.mean():6.1f} +-{pay.std()/G**.5:.1f} c:',' '.join(f'{x:4.1f}' for x in C.mean((0,1))[::3].tolist()),' p/pl:',round(P.mean((0,1)).sum().item(),1),flush=True)
run('v4 base (v3 gp, const net 25)',rule())
for r0 in [0.6,1.4]: run(f'round0 x{r0}',rule(r0=r0))
for d,dF in [(6,5),(6,10),(10,10),(10,20)]: run(f'drop net D{d} fine {dF}',rule(drop=d,dropF=dF))
for tg,tE in [(14,16),(15,16),(15,20)]: run(f'group trigger <{tg} to {tE}',rule(trig=tg,trigE=tE))
for F,gm,w in [(35,2.3,8),(37,2.3,8),(30,2.0,6),(25,1.9,6),(33,2.5,7)]: run(f'F{F} gm{gm} w{w}',rule(F=F,gm=gm,w=w))
