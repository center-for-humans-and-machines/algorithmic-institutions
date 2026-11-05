# v5: is the opponent-dependence of mid's effect real? pooled quadratic surface over v3+v4 points (all 4 opponents stacked, opponent fixed effects) plus mid x harsh-opponent and decay x harsh interactions, with point-level se
import json, numpy as np
P=[]
for v in ['pf-20261106-11-v3','pf-20261106-11-v4']:
    P+=json.load(open(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/{v}_sweep/sweep.json'))['points']
rows=[];y=[];w=[]
for p in P:
    a=p['params']; x=[np.log(a['rate'])-2,a['delta']+1,a['decay']-0.5,a['mid']-1]
    q=[x[i]*x[j] for i in range(4) for j in range(i,4)]
    for k in range(1,5):
        o=[float(k==j) for j in range(1,5)]; harsh=float(k in (1,3))
        rows.append(o+x+q+[harsh*x[3],harsh*x[2],harsh*x[0],harsh*x[1]]); y.append(p['pools'][f'opponent_{k}']['pool']); w.append(1/p['pools'][f'opponent_{k}']['pool_se']**2)
X=np.array(rows);y=np.array(y);W=np.sqrt(np.array(w))
b,res,_,_=np.linalg.lstsq(X*W[:,None],y*W,rcond=None)
r=y-X@b; s2=(r**2*W**2).sum()/(len(y)-X.shape[1])
cov=s2*np.linalg.inv((X*W[:,None]).T@(X*W[:,None]))
se=np.sqrt(np.diag(cov))
for name,i in [('mid x harsh',-4),('decay x harsh',-3),('lograte x harsh',-2),('delta x harsh',-1),('mid',7),('decay',6)]:
    print(f'{name}: {b[i]:+.2f} (se {se[i]:.2f})')
