"""Random search of fine = F*(shortfall/target)^gam for rounds < w, surrogate v2; quadratic fit of the surface."""
import torch as th, numpy as np, sys, json
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from harness import run
def mk(F,tg,gam,w):
    return lambda C,P,t: F*((tg-C[...,-1]).clamp(min=0)/tg)**gam*float(t<w)
rng=np.random.default_rng(7); res=[]
for i in range(160):
    p=dict(F=rng.uniform(16,40),tg=int(rng.integers(12,21)),gam=float(np.exp(rng.uniform(np.log(.6),np.log(2.6)))),w=int(rng.integers(5,16)))
    res.append(dict(m=run(mk(**p),G=2500,seed=6),**p))
import pandas as pd
d=pd.DataFrame(res).sort_values('m',ascending=False); print(d.head(20).round(2))
json.dump(res,open('/tmp/pf7_rp.json','w'))
from sklearn.preprocessing import PolynomialFeatures; from sklearn.linear_model import Ridge
X=d[['F','tg','gam','w']].values.copy(); X[:,2]=np.log(X[:,2])
pf=PolynomialFeatures(2); m=Ridge(1e-3).fit(pf.fit_transform(X),d.m)
grid=np.array(np.meshgrid(np.linspace(16,40,25),np.arange(12,21),np.log(np.linspace(.6,2.6,21)),np.arange(5,16))).reshape(4,-1).T
pr=m.predict(pf.transform(grid)); i=np.argmax(pr); print('quad peak',grid[i],np.exp(grid[i,2]),pr[i])
