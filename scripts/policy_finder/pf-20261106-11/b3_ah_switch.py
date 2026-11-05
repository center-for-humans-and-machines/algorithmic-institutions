# v2: how the artificial humans switch, from the v1 sweep's games (both groups, all managers)
import numpy as np, pandas as pd, sys
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from b2_arrays import load
d = load(); c=d['c'].astype(float); p=d['p'].astype(float); g=d['g']; v=d['v']
rows=[]
for t in [3,7,11,15,19]:
    gi = g[:,:,t]
    same = (g[:,:,t-3:t+1]==gi[:,:,None]).all(-1)
    n_own = (g[:,:,t][:,None,:]==gi[:,:,None]).sum(-1)
    def grp_mean(x, own=True):
        m = (g[:,:,t][:,None,:]==gi[:,:,None]) if own else (g[:,:,t][:,None,:]!=gi[:,:,None])
        return (m*x[:,None,:]).sum(-1)/np.maximum(m.sum(-1),1), m.sum(-1)
    # block-mean payoff proxy per head: 1.6*mean c - mean p of group, averaged over the block
    sh_own=0; sh_oth=0
    for r in range(t-3,t+1):
        gr = g[:,:,r]
        mo = (gr[:,None,:]==gr[:,:,None]); mx = ~mo
        sh_own += (mo*(1.6*c[:,None,:,r])).sum(-1)/np.maximum(mo.sum(-1),1)
        sh_oth += np.where(mx.sum(-1)>0,(mx*(1.6*c[:,None,:,r])).sum(-1)/np.maximum(mx.sum(-1),1), np.nan)
    pm_own = p[:,:,t-3:t+1].mean(-1)
    # mean punishment in the other group over the block (of its members)
    po=0; cnt=0
    for r in range(t-3,t+1):
        gr=g[:,:,r]; mx=(gr[:,None,:]!=gr[:,:,None])
        po += (mx*p[:,None,:,r]).sum(-1); cnt += mx.sum(-1)
    po = np.where(cnt>0, po/np.maximum(cnt,1), np.nan)
    rows.append(pd.DataFrame(dict(t=t, sw=(g[:,:,t+1]!=gi).ravel(), pb=pm_own.ravel(), plast=p[:,:,t].ravel(),
        cown=c[:,:,t-3:t+1].mean(-1).ravel(), gap=((sh_oth-sh_own)/4).ravel(), n=n_own.ravel(), pother=po.ravel(), stay=same.ravel())))
X = pd.concat(rows, ignore_index=True)
print('switch rate', X.sw.mean().round(3), X.groupby('t').sw.mean().round(3).to_dict())
print(X.groupby(pd.cut(X.pb,[-.1,0,1,3,6,10,30]),observed=True).sw.agg(['mean','size']).round(3))
print(X.groupby(pd.cut(X.plast,[-.1,0,2,5,10,30]),observed=True).sw.agg(['mean','size']).round(3))
print(X.groupby(pd.cut(X.gap,[-30,-10,-5,-2,2,5,10,30]),observed=True).sw.agg(['mean','size']).round(3))
print(X.groupby('n').sw.agg(['mean','size']).round(3))
print(X.groupby(pd.cut(X.pother,[-.1,0,1,3,6,30]),observed=True).sw.agg(['mean','size']).round(3))
import statsmodels.formula.api as smf
Y = X.dropna().sample(400000, random_state=0)
Y['sw']=Y.sw.astype(int)
m = smf.logit('sw ~ pb + plast + gap + n + pother + cown + C(t)', data=Y).fit(disp=0)
print(m.params.round(4)); print(m.bse.round(4))
