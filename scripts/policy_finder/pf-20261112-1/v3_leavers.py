# v3: who leaves an unpunishing group? Decision rows of the rule's group at v2's zero-equivalent points
# (bar < 0 or first_dec == 5), features the rule itself can compute.
import pandas as pd, numpy as np, glob, sys, json
sys.path.insert(0,'scripts/policy_finder/pf-20261112-1'); from v3_load import load
sw=json.load(open('plots/simulation/policy_finder/pf-20261112-1-v2_sweep/sweep.json'))
zero={p['name'] for p in sw['points'] if p['params']['bar']<0 or p['params']['first_dec']==5}
rows=[]
for f in sorted(glob.glob('plots/simulation/policy_finder/pf-20261112-1-v2_sweep/per_round_p*.parquet')):
    D=load(f); keep=np.isin(D['point'],list(zero))
    P,C,G,V=(D[k][keep].astype(float) for k in 'PCGV'); M=D['mine'][keep][:,None]
    del D
    for dec in [3,7,11,15,19]:
        w=slice(dec-3,dec+1)
        mine=(G[:,:,dec]==M); oth=~mine
        n_m=mine.sum(1,keepdims=True); n_o=oth.sum(1,keepdims=True)
        cm=(C[:,:,dec]*mine).sum(1,keepdims=True)/np.maximum(n_m,1)
        co=(C[:,:,dec]*oth).sum(1,keepdims=True)/np.maximum(n_o,1)
        po=(P[:,:,dec]*oth).sum(1,keepdims=True)/np.maximum(n_o,1)
        pow_=((P[:,:,w]*(G[:,:,w]!=M[...,None])).sum((1,2))/np.maximum((G[:,:,w]!=M[...,None]).sum((1,2)),1))[:,None]
        cmw=((C[:,:,w]*(G[:,:,w]==M[...,None])).sum((1,2))/np.maximum((G[:,:,w]==M[...,None]).sum((1,2)),1))[:,None]
        cow=((C[:,:,w]*(G[:,:,w]!=M[...,None])).sum((1,2))/np.maximum((G[:,:,w]!=M[...,None]).sum((1,2)),1))[:,None]
        own_w=C[:,:,w].mean(-1)
        new=(G[:,:,dec-4]!=M) if dec>3 else np.zeros_like(mine)
        sel=mine
        b=lambda x: np.broadcast_to(x,mine.shape)[sel]
        rows.append(pd.DataFrame({'dec':dec,'c':C[:,:,dec][sel],'cw':own_w[sel],'dc':(C[:,:,dec]-C[:,:,dec-1])[sel],
            'cm':b(cm),'co':b(co),'cmw':b(cmw),'cow':b(cow),'po':b(po),'pow':b(pow_),'n':b(n_m),'valid':V[:,:,dec][sel],
            'new':new[sel].astype(int),'leave':(G[:,:,dec+1]!=M)[sel].astype(int)}))
r=pd.concat(rows); print(len(r), r.leave.mean())
r.to_parquet('/tmp/v3_leavers.parquet')
print(r.groupby('dec').leave.mean().round(3))
r['rel']=r.c-r.cm
for col,bins in [('c',[-1,0,2,5,9,14,19,20]),('rel',[-21,-8,-4,-1,1,4,8,21]),('n',[0,1,2,3,4,5,6,7,8]),('pow',[-1,0.5,1.5,3,6,31]),('dc',[-21,-5,-1,0,1,5,21])]:
    print(r.groupby(pd.cut(r[col],bins),observed=True).leave.agg(['mean','size']).round(3).T)
import statsmodels.formula.api as smf
m=smf.logit('leave ~ c + cw + rel + cmw + cow + pow + po + n + new + valid + dec',r.sample(400000,random_state=0).assign(dec=lambda x:x.dec.astype(str))).fit(disp=0)
print(m.params.round(3)); print(m.prsquared)
r['ph']=m.predict(r.assign(dec=r.dec.astype(str)))
print(r.groupby(pd.qcut(r.ph,10)).leave.mean().round(3))
