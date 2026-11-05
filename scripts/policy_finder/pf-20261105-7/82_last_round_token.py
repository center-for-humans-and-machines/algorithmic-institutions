"""Real sweeps v2-v6: players at c 15-19 in the LAST opening round (t = window-1), whose fine there is
1-3 vs 0 (variation from the rule's params), and who are unfined afterwards for 3 rounds.
Outcome: own mean c over t+1..t+8 minus c_t; peers' mean change likewise. Regression with controls
(c_t dummies, others' mean at t, own c at t-1, round dummies, version dummies)."""
import numpy as np, json
VS=['pf-20261105-7-v2','pf-20261105-7-v3','pf-20261105-7-v4','pf-20261105-7-v5','pf-20261105-7-v6']
rows=[]
for vi,v in enumerate(VS):
    d=np.load(f'/tmp/pf7/{v}_sweep.npz'); C=d['C'].astype(float);P=d['P'].astype(float);V=d['V'].astype(bool);R=d['R']
    prm={p['name']:p['params'] for p in json.load(open(f'plots/simulation/policy_finder/{v}_sweep/sweep.json'))['points']}
    W=np.array([prm[r]['window'] for r in R])
    for w in np.unique(W):
        t=w-1
        if t<2 or t+8>23: continue
        g=np.nonzero(W==w)[0]
        c=C[g];p=P[g];val=V[g]
        for i in range(4):
            ok=(c[:,i,t]>=15)&(c[:,i,t]<=19)&val[:,i,t]&(p[:,i,t]<=3)&(p[:,i,t+1:t+4].sum(1)==0)&val[:,i,t+1:t+9].all(1)
            for k in np.nonzero(ok)[0]:
                oth=[j for j in range(4) if j!=i]
                rows.append([vi,t,c[k,i,t],c[k,i,t-1],c[k,oth,t].mean(),p[k,i,t],c[k,i,t+1:t+9].mean()-c[k,i,t],c[k,oth,t+1:t+9].mean()-c[k,oth,t].mean(),p[k,oth,t].sum(), p[k,i,:t].sum()])
A=np.array(rows); print('n',len(A))
f=A[:,5]
for lab,m in [('0',f==0),('1',f==1),('2-3',f>=2)]:
    print(lab,m.sum(),'own %.2f peers %.2f c_t %.2f others %.2f ownpast fines %.1f'%(A[m,6].mean(),A[m,7].mean(),A[m,2].mean(),A[m,4].mean(),A[m,9].mean()))
def reg(y,cols):
    X=[np.ones(len(A))]
    for cv in range(15,20): X.append((A[:,2]==cv).astype(float))
    X=X[:-1]
    for vi in range(1,5): X.append((A[:,0]==vi).astype(float))
    for tt in np.unique(A[:,1])[1:]: X.append((A[:,1]==tt).astype(float))
    X+= [A[:,3],A[:,4],A[:,8],A[:,9]]
    X+= cols
    X=np.column_stack(X); b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b
    cov=(r@r/(len(y)-X.shape[1]))*np.linalg.inv(X.T@X); return b[-len(cols):],np.sqrt(np.diag(cov))[-len(cols):]
for nm,yy in [('own',A[:,6]),('peers',A[:,7])]:
    b,s=reg(yy,[(f==1).astype(float),(f>=2).astype(float)])
    print(nm,'fine1 %.2f (%.2f)  fine2-3 %.2f (%.2f)'%(b[0],s[0],b[1],s[1]))
