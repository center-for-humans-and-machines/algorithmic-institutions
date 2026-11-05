"""Nudge candidates on sur3_all3 (common seeds, two seed sets): v3/v4-like base (opening F 30, gm 2.2,
window 7, net c<=2 at 22 to round 16) plus a token fine for players short of 20.
nudge(k, lo, a, b): fine k for c in [lo,19] in rounds a..b-1 (on top of the base, max with base)."""
import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/'+(sys.argv[1] if len(sys.argv)>1 else 'sur3_all3.pt'),weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v4_sweep.npz')['C'][:,:,0]).float()
G=3000
def rule(F=30,gm=2.2,w=7,floor=2,E=16,nf=22.,k=0,lo=16,a=0,b=0,hi=19):
    def r(C,P,t):
        c=C[...,-1]
        base=F*((20-c).clamp(min=0)/20)**gm if t<w else nf*(c<=floor).float()*float(t<E)
        nud=k*((c>=lo)&(c<=hi)).float()*float(a<=t<b)
        return th.maximum(base,nud)
    return r
res={}
for s in [5,6]:
    C0=C0all[th.randint(0,C0all.shape[0],(G,),generator=th.Generator().manual_seed(s))]
    def run(label,r):
        pay,C,P=simulate(net,r,C0,seed=11+s,ret_hist=True)
        res.setdefault(label,[]).append(pay.mean().item())
        print(f'seed{s} {label:34s} {pay.mean():6.1f} c:',' '.join(f'{x:4.1f}' for x in C.mean((0,1))[::3].tolist()),' p/pl:',round(P.mean((0,1)).sum().item(),1),flush=True)
    run('base',rule())
    for k in [1,2,3]:
        for lo in [16,18]:
            for a,b in [(0,7),(0,16),(7,16),(0,23)]:
                run(f'k{k} lo{lo} r{a}-{b}',rule(k=k,lo=lo,a=a,b=b))
    for k in [1,2]: run(f'k{k} all c<20 r0-16',rule(k=k,lo=0,a=0,b=16))
print('--- mean over seeds, diff to base')
for l,v in res.items(): print(f'{l:34s} {np.mean(v):6.1f} {np.mean(v)-np.mean(res["base"]):+5.1f}')
