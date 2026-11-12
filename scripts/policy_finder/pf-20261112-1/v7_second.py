# v7: causal value of a second target: v4's points randomize n_push (1-3). For v4 touched games, at the first push
# round, the second-lowest window-mean member's gap (others' mean excl. him minus his mean) and whether he is under
# the point's bar; compare dlead by n_push within gap bins of that second member (pre-treatment state).
import numpy as np, pandas as pd, json
Z=pd.read_parquet('/tmp/v6_Z.parquet')
pars={p['name']:p['params'] for p in json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261112-1-v4_sweep/sweep.json'))['points']}
rows=[]
for part in '12':
    d=np.load(f'/tmp/v6_v4_{part}.npz',allow_pickle=True)
    key,seat,pun,P,G,C,V,lead,point=d['key'],d['seat'],d['pun'],d['P'],d['G'],d['C'],d['V'],d['lead'],d['point']
    idx=np.where(pun)[0]
    m=pd.DataFrame(dict(key=key[idx],seat=seat[idx],i=idx)).merge(Z,on=['key','seat'],suffixes=('','_z'))
    for i,lz in zip(m.i.values,m.lead.values):
        pr=P[i]; rr=np.where(pr.sum(0)>0)[0][0]
        if rr>15: continue
        own=(G[i]==seat[i])[:,rr]; cw=C[i][:,rr-3:rr+1].astype(float); vw=V[i][:,rr-3:rr+1]
        cm=(cw*vw).sum(1)/np.maximum(vw.sum(1),1); ok=own&(vw.sum(1)>0)&(V[i][:,rr]>0)
        order=[j for j in np.argsort(cm+1e-3*np.arange(8)) if ok[j]]
        if len(order)<3: continue
        gaps=[(cm[ok].sum()-cm[j])/(ok.sum()-1)-cm[j] for j in order[:3]]
        pp=pars[point[i]]
        rows.append(dict(dec=rr,size=own.sum(),n_push=pp['n_push'],bar=pp['bar'],amount=pp['amount'],cm1=cm[order[0]],cm2=cm[order[1]],cm3=cm[order[2]],
                         g1=gaps[0],g2=gaps[1],g3=gaps[2],nt=(pr[:,rr]>0).sum(),dlead=lead[i]-lz))
df=pd.DataFrame(rows); df.to_parquet('/tmp/v7_second.parquet'); print(len(df))
e=df[(df.cm2<=df.bar)]   # second member eligible: n_push decides whether he is punished
e=e.assign(g2b=pd.cut(e.g2,[-20,2,4,6,8,10,20]),g1b=pd.cut(e.g1,[-20,4,6,8,10,20]))
print(e.pivot_table(index='g2b',columns='n_push',values='dlead',aggfunc=['mean','count'],observed=True).round(2))
print(e.pivot_table(index='g1b',columns='n_push',values='dlead',aggfunc=['mean','count'],observed=True).round(2))
f=e[e.n_push<=2]; 
import statsmodels.formula.api as smf
print(smf.ols('dlead~C(n_push)*g2+g1+C(dec)+amount',f).fit(cov_type='HC1').summary().tables[1])
