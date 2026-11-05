"""Opening schedule F*(s)^k evaluated at c=0,5,8,10,12,14,16 for the best real sweep points of v3-v5
(by pool) and the worst, to see which part of the schedule the ridge F<->curvature pins down."""
import json, numpy as np
rows=[]
for v in ['pf-20261105-7-v3','pf-20261105-7-v4','pf-20261105-7-v5']:
    d=json.load(open(f'plots/simulation/policy_finder/{v}_sweep/sweep.json'))
    for p in d['points']:
        q=p['params']; rows.append((p['pool'],q['free_rider_fine'],q['curvature'],q['window'],v[-2:]))
rows.sort(reverse=True)
cs=[0,5,8,10,12,14,16]
f=lambda F,k,c: np.floor(min(30,F*((20-c)/20)**k))
print('pool  F  curv  win  fines at c=',cs)
for r in rows[:25]+[None]+rows[len(rows)//2-5:len(rows)//2+5]:
    if r is None: print('--- median'); continue
    print('%.1f %5.1f %4.2f %2d %s'%r[:4]%() if False else '%.1f %5.1f %4.2f %2d %s'%(r[0],r[1],r[2],r[3],r[4]), [f(r[1],r[2],c) for c in cs])
# correlation of pool with fine at each c among points with window 5-9
import numpy as np
A=np.array([[r[0]]+[min(30,r[1]*((20-c)/20)**r[2]) for c in cs] for r in rows if 5<=r[3]<=9])
print('corr(pool, fine at c) window 5-9:', [round(np.corrcoef(A[:,0],A[:,i+1])[0,1],2) for i in range(len(cs))])
for i,c in enumerate(cs):
    x=A[:,i+1]; qs=np.quantile(x,[0,.2,.4,.6,.8,1])
    print('c=%2d'%c, ' '.join('[%.1f-%.1f]:%.1f'%(qs[j],qs[j+1],A[(x>=qs[j])&(x<=qs[j+1]),0].mean()) for j in range(5)))
