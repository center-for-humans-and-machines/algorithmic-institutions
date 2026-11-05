"""Gate gain on a second surrogate (sur3_all: trained on v1+v2 only, never saw v3's net) vs sur3_all3, 3 seed sets x 3000."""
import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import Net, simulate
th.set_num_threads(2)
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v4_sweep.npz')['C'][:,:,0]).float()
def rule(F=30,gm=2.2,w=7,floor=2,E=16,nf=22.,m=None):
    def r(C,P,t):
        c=C[...,-1]; s=F*((20-c).clamp(min=0)/20)**gm
        if t<w: return s if m is None else s*(c<c.mean(1,keepdim=True)+m).float()
        return nf*(c<=floor).float()*float(t<E)
    return r
cands=[('base',{}),('m1',dict(m=1)),('m3',dict(m=3)),('m6',dict(m=6)),('F35 ungated',dict(F=35)),('F40 m1',dict(F=40,m=1)),('F40 m-1',dict(F=40,m=-1)),('F44 gm2.3 m-1.9 nf28',dict(F=44,gm=2.3,m=-1.9,nf=28)),('F33 gm2.2 w5 m2.6 nf25',dict(F=33,gm=2.2,w=5,m=2.6,E=16,nf=25,floor=1))]
for f in ['sur3_all.pt','sur3_all3.pt']:
    net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/'+f,weights_only=True)); net.eval()
    res={}
    for s in [41,42,43]:
        C0=C0all[th.randint(0,C0all.shape[0],(3000,),generator=th.Generator().manual_seed(s))]
        for l,kw in cands: res.setdefault(l,[]).append(simulate(net,rule(**kw),C0,seed=s).mean().item())
    b=np.array(res['base'])
    print(f, ' | '.join(f'{l} {np.mean(v):.1f} ({np.mean(np.array(v)-b):+.1f})' for l,v in res.items()),flush=True)
