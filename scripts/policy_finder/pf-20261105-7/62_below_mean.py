"""Gate the opening schedule on the group: fine only players with c < (group mean incl. self) + m
(or the others' mean + m), on sur3_all3, three seed sets x 3000 games. Base as in 60."""
import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur3_all3.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v4_sweep.npz')['C'][:,:,0]).float()
G=3000
def rule(F=30,gm=2.2,w=7,floor=2,E=16,nf=22.,m=None,ref='mean'):
    def r(C,P,t):
        c=C[...,-1]
        sched=F*((20-c).clamp(min=0)/20)**gm
        if t<w:
            if m is None: return sched
            if ref=='mean': g=c.mean(1,keepdim=True)
            elif ref=='others': g=(c.sum(1,keepdim=True)-c)/3
            elif ref=='max': g=c.max(1,keepdim=True).values
            return sched*(c<g+m).float()
        return nf*(c<=floor).float()*float(t<E)
    return r
res={}
seeds=[5,6,8]
for s in seeds:
    C0=C0all[th.randint(0,C0all.shape[0],(G,),generator=th.Generator().manual_seed(s))]
    def run(label,r):
        pay,C,P=simulate(net,r,C0,seed=11+s,ret_hist=True)
        res.setdefault(label,[]).append((pay.mean().item(),P.mean((0,1)).sum().item(),C.mean().item()))
    run('base',rule())
    for m in [-3,-1,0,1,3]: run(f'mean m{m}',rule(m=m))
    for m in [-2,0,2]: run(f'others m{m}',rule(m=m,ref='others'))
    for m in [-6,-3]: run(f'max m{m}',rule(m=m,ref='max'))
    for F,gm,w in [(35,2.2,7),(40,2.2,7),(30,1.8,7),(35,1.8,7),(30,2.2,9),(30,2.6,7),(40,2.6,7)]:
        run(f'mean m0 F{F} gm{gm} w{w}',rule(F=F,gm=gm,w=w,m=0))
        run(f'nogate  F{F} gm{gm} w{w}',rule(F=F,gm=gm,w=w))
b=np.array(res['base'])
for l,v in res.items():
    v=np.array(v); d=v[:,0]-b[:,0]
    print(f'{l:30s} {v[:,0].mean():6.1f} {d.mean():+5.1f} (sd {d.std():.1f})  p/pl {v[:,1].mean():5.1f}  meanc {v[:,2].mean():5.2f}')
