# v2: how the artificial humans' contributions respond to punishment (v1 sweep games)
import numpy as np, pandas as pd, sys
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from b2_arrays import load
d = load(); c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); g=d['g']; v=d['v']
rows=[]
rng=np.random.default_rng(0); sel=rng.choice(len(c), 40000, replace=False)
c=c[sel];p=p[sel];g=g[sel];v=v[sel]
for t in range(0,22):
    gi=g[:,:,t]; mo=(gi[:,None,:]==gi[:,:,None]); np.einsum('gii->gi',mo)[:]=False
    oth = (mo*c[:,None,:,t]).sum(-1)/np.maximum(mo.sum(-1),1)
    stay = (g[:,:,t+1]==gi)&v[:,:,t]&v[:,:,t+1]
    rows.append(pd.DataFrame(dict(t=t, c=c[:,:,t].ravel(), p=p[:,:,t].ravel(), oth=oth.ravel(), n=mo.sum(-1).ravel()+1,
       dc1=(c[:,:,t+1]-c[:,:,t]).ravel(), dc2=(c[:,:,min(t+2,23)]-c[:,:,t]).ravel(), stay=stay.ravel(),
       stay2=(stay&(g[:,:,min(t+2,23)]==gi)).ravel(), pprev=(p[:,:,t-1] if t>0 else 0*p[:,:,0]).ravel())))
X=pd.concat(rows,ignore_index=True); X=X[X.stay]
X['cb']=pd.cut(X.c,[-1,0,4,8,12,16,19,20]); X['pb']=pd.cut(X.p,[-1,0,2,5,10,20,30])
print('next-round change dc1 by c bin x p bin'); print(X.pivot_table(index='cb',columns='pb',values='dc1',aggfunc='mean',observed=True).round(2))
print(X.pivot_table(index='cb',columns='pb',values='dc1',aggfunc='size',observed=True))
X2=X[X.stay2]; print('two-round change dc2'); print(X2.pivot_table(index='cb',columns='pb',values='dc2',aggfunc='mean',observed=True).round(2))
X['rel']=pd.cut(X.c-X.oth,[-20,-8,-4,-1,1,4,8,20])
print('dc1 by c-others x p bin'); print(X.pivot_table(index='rel',columns='pb',values='dc1',aggfunc='mean',observed=True).round(2))
print('dc1 for 8<c<=16 by others mean x p'); Y=X[(X.c>8)&(X.c<=16)]
print(Y.pivot_table(index=pd.cut(Y.oth,[-1,6,10,14,20]),columns='pb',values='dc1',aggfunc='mean',observed=True).round(2))
import statsmodels.formula.api as smf
m=smf.ols('dc1 ~ c + oth + p + I(p**2) + p:c + p:oth + pprev', data=X.sample(500000,random_state=0)).fit()
print(m.params.round(4))
