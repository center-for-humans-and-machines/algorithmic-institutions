"""Random search over a flexible family on surrogate v2."""
import torch as th, numpy as np, sys, json
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from harness import run
def fam(tg,sl,w,gam,rise,esc,rel):
    def r(C,P,t):
        c=C[...,-1]
        if t>=w: return th.zeros_like(c)
        others=(c.sum(1,keepdim=True)-c)/3
        T=(tg+rise*t)*(1-rel)+rel*(others+tg-12)
        T=T.clamp(max=20)
        sh=(T-c).clamp(min=0)
        npun=(P>0).float().sum(-1) if P.shape[-1]>0 else th.zeros_like(c)
        return sl*14*(sh/14)**gam*(1+esc*npun)
    return r
rng=np.random.default_rng(0); res=[]
base=dict(tg=14,sl=2.08,w=9,gam=1.,rise=0.,esc=0.,rel=0.)
print('base',run(fam(**base),G=2000))
for i in range(400):
    p=dict(tg=int(rng.integers(10,21)),sl=float(np.exp(rng.uniform(np.log(.8),np.log(4)))),w=int(rng.integers(4,17)),
           gam=float(np.exp(rng.uniform(np.log(.5),np.log(2.5)))),rise=float(rng.uniform(-.3,.4)),esc=float(rng.uniform(-.15,.3)),rel=float(rng.uniform(0,1)))
    m=run(fam(**p),G=1500,seed=1)
    res.append((m,p))
res.sort(key=lambda x:-x[0])
for m,p in res[:25]: print(round(m,1),{k:round(v,2) for k,v in p.items()})
json.dump(res,open('/tmp/pf7_search.json','w'))
# marginal: correlation of params with score
import pandas as pd
d=pd.DataFrame([dict(m=m,**p) for m,p in res]); print(d.corr()['m'].round(2))
