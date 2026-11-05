# v6: as j5, at block starts 4, 8, 12: within size bins at round 4b, future pool (rounds 4b..23) on log(decay), controls; v3+v4+v5 pooled with sweep dummies
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
D=[]
for tag in ['v3','v4','v5']:
    d = load(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-{tag}_sweep/', tag)
    D.append((tag,d))
bins=[(1,1),(2,3),(4,5),(6,7),(8,8)]
for t0 in [4,8,12]:
    print(f'=== block start r{t0}')
    for lo,hi in bins:
        Xs=[];ys=[]
        for j,(tag,d) in enumerate(D):
            c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']
            fut=(1.6*c*m - p*m).sum(1)[:,t0:].mean(1)
            n0=m[:,:,t0].sum(1)
            P=d['params'].set_index(d['params'].name.str.extract(r'_s(\d+)')[0].astype(float).fillna(-1).astype(int))
            pt=d['pt']
            def col(k): return P[k].reindex(pt).to_numpy() if k in P else np.zeros(len(pt))
            s=(pt>=0)&(n0>=lo)&(n0<=hi)
            ld=np.log(np.maximum(col('decay'),0.02))
            O=np.stack([d['opp']==k for k in [2,3,4]],1).astype(float)
            S=np.zeros((len(pt),3)); S[:,j]=1
            X=np.column_stack([S,ld,np.log(col('rate')),col('delta'),col('mid'),col('kappa'),O])[s]
            Xs.append(X); ys.append(fut[s])
        X=np.concatenate(Xs); y=np.concatenate(ys)
        b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b; cov=np.linalg.pinv(X.T@X)*r.var()
        print(f'  size {lo}-{hi}: n={len(y):6d} fut {y.mean():6.1f}  dlog(decay) {b[3]:+.2f} ({np.sqrt(cov[3,3]):.2f})')
