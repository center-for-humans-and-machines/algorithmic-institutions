"""v4 candidates on the surrogate trained on all three sweeps (sur3_all3), common seeds, differences only.
Reference: v3's final peak. Variants: constant net fine, a second (mid) net tier, an opening taper."""
import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
tag=sys.argv[1] if len(sys.argv)>1 else 'all3'
net=Net(); net.load_state_dict(th.load(f'scripts/policy_finder/pf-20261105-7/sur3_{tag}.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v3_sweep.npz')['C'][:,:,0]).float()
G=int(sys.argv[2]) if len(sys.argv)>2 else 3000
C0=C0all[th.randint(0,C0all.shape[0],(G,),generator=th.Generator().manual_seed(5))]
def rule(F=29.7,gm=2.18,w=7,floor=2,E=16,net=None,scale=0.85,taper=0.0,mid=None,midE=None,midF=0):
    def r(C,P,t):
        c=C[...,-1]; f=F*((20-c).clamp(min=0)/20)**gm
        if t<w: return f*(1-taper*t/max(w-1,1))
        out=th.zeros_like(c)
        if t<E:
            nf=f*scale if net is None else th.full_like(c,float(net))
            out=out+nf*(c<=floor).float()
        if mid is not None and t<midE:
            out=out+midF*((c>floor)&(c<=mid)).float()
        return out
    return r
def run(label,r):
    pay,C,P=simulate(net,r,C0,seed=11,ret_hist=True)
    print(f'{label:40s} {pay.mean():6.1f} +-{pay.std()/G**.5:.1f} c:',' '.join(f'{x:4.1f}' for x in C.mean((0,1))[::3].tolist()),' p/pl:',round(P.mean((0,1)).sum().item(),1),flush=True)
    return pay.mean().item()
run('zero',lambda C,P,t: th.zeros_like(C[...,-1]))
run('v3 gp',rule())
run('v3 s229',rule(25.76,1.88,5,3,18,scale=0.67))
run('no net',rule(E=0))
for nf in [15,20,25,30]: run(f'const net {nf}',rule(net=nf))
for nf,fl in [(30,1),(30,3),(30,4)]: run(f'const net {nf} floor {fl}',rule(net=nf,floor=fl))
for E in [13,18,20]: run(f'const net 30 E{E}',rule(net=30,E=E))
for tp in [0.3,0.6]: run(f'taper {tp}',rule(taper=tp))
for tp,w in [(0.5,9),(0.7,10)]: run(f'taper {tp} w{w}',rule(taper=tp,w=w))
for mid,mE,mF in [(6,10,8),(6,12,5),(8,10,5),(6,10,15)]: run(f'mid tier c<={mid} to {mE} fine {mF}',rule(mid=mid,midE=mE,midF=mF))
