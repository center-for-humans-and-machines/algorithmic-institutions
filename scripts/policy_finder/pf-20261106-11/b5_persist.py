# v2: how long a punishment's contribution gain lasts in the artificial humans, and its pool value
import numpy as np, pandas as pd, sys
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from b2_arrays import load
import statsmodels.api as sm
d = load(); rng=np.random.default_rng(1); sel=rng.choice(len(d['c']),30000,replace=False)
c=d['c'][sel].astype(float); p=d['p'][sel].astype(float); g=d['g'][sel]; v=d['v'][sel]
for k in [1,2,3,4,6,8]:
    rows=[]
    for t in range(1,24-k):
        gi=g[:,:,t]; mo=(gi[:,None,:]==gi[:,:,None]); np.einsum('gii->gi',mo)[:]=False
        oth=(mo*c[:,None,:,t]).sum(-1)/np.maximum(mo.sum(-1),1)
        stay=(g[:,:,t:t+k+1]==gi[:,:,None]).all(-1)&v[:,:,t]&v[:,:,t+k]
        low=(c[:,:,t]<=oth)
        # later punishments in the window, to control
        plater=p[:,:,t+1:t+k].sum(-1) if k>1 else 0*c[:,:,t]
        m=stay&low
        rows.append(np.stack([c[:,:,t+k][m]-c[:,:,t][m], c[:,:,t][m], oth[m], p[:,:,t][m], plater[m], c[:,:,t-1][m], p[:,:,t-1][m], np.full(m.sum(),t)],1))
    A=np.concatenate(rows); X=sm.add_constant(A[:,1:]); r=sm.OLS(A[:,0],X).fit()
    print('k',k,'n',len(A),'coef p_t %.3f  p_later %.3f  c %.3f oth %.3f c_prev %.3f p_prev %.3f t %.3f'%(r.params[3],r.params[4],r.params[1],r.params[2],r.params[5],r.params[6],r.params[7]))
