# v3: do newcomers to my group (joined at this block) react differently to punishment, in exits and contributions?
import numpy as np, pandas as pd, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
import statsmodels.formula.api as smf
d = load()
rng = np.random.default_rng(2); idx = rng.choice(len(d['c']), 40000, replace=False)
c=d['c'][idx].astype(float); p=d['p'][idx].astype(float); m=d['mine'][idx]
rows=[]
for t in [7,11,15,19]:
    blk=slice(t-3,t+1); inb=m[:,:,t]&m[:,:,t-3]; new=m[:,:,t-3]&~m[:,:,t-4]
    tot=(c[:,:,t-3]*m[:,:,t-3]).sum(1,keepdims=True); n=m[:,:,t-3].sum(1,keepdims=True)
    om=(tot-c[:,:,t-3])/np.maximum(n-1,1)
    rows.append(pd.DataFrame(dict(t=t,new=new[inb].astype(int),p0=p[:,:,t-3][inb],pm=p[:,:,blk].mean(2)[inb],c0=c[:,:,t-3][inb],gap0=(c[:,:,t-3]-om)[inb],dc=(c[:,:,t-2]-c[:,:,t-3])[inb],leave=(~m[:,:,t+1])[inb].astype(float))))
x=pd.concat(rows)
print('share new', x.new.mean().round(3))
print(smf.logit('leave ~ pm*new + c0 + C(t)', x).fit(disp=0).params.round(3))
print(smf.ols('dc ~ p0*new + gap0 + c0', x[x.gap0<-2]).fit().params.round(3))
