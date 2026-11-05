"""As 70, but the opening schedule in coordinates (mid_fine = fine at c=10 = F*0.5^curv, curvature).
Fit on v3-v5 sweeps+finals (or the versions on argv); peak, profiles, and a curvature x mid_fine grid."""
import json, sys, numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
VERS=sys.argv[1:] or ['pf-20261105-7-v3','pf-20261105-7-v4','pf-20261105-7-v5']
K=['mid_fine','curvature','window','gate_margin','net_floor','net_end','net_fine']
def conv(p):
    q=dict(p)
    if 'net_scale' in q: q['net_fine']=min(30.,q['free_rider_fine']*q.pop('net_scale'))
    q.setdefault('gate_margin',8.); q['gate_margin']=min(q['gate_margin'],8.)
    if q['net_end']<=q['window']: q['net_end']=q['window']; q['net_fine']=0.
    if 'mid_fine' not in q: q['mid_fine']=q['free_rider_fine']*0.5**q['curvature']
    return [q[k] for k in K]
X=[];Y=[];S=[]
for v in VERS:
    for f in ['sweep','final/sweep']:
        for p in json.load(open(f'plots/simulation/policy_finder/{v}_sweep/{f}.json'))['points']:
            X.append(conv(p['params']));Y.append(p['pool']);S.append(p['pool_se'])
X=np.array(X,float);Y=np.array(Y);S=np.array(S)
def tr(X): Z=X.copy(); Z[:,0]=np.log(Z[:,0]); Z[:,1]=np.log(Z[:,1]); return Z
Z=tr(X); mu=Z.mean(0); sd=Z.std(0)
gp=GaussianProcessRegressor(Ck(1.)*Matern(length_scale=np.ones(7),nu=2.5)+WhiteKernel(0.01),alpha=(S/Y.std())**2,normalize_y=True,n_restarts_optimizer=3,random_state=0)
gp.fit((Z-mu)/sd,Y); print(gp.kernel_)
def pred(X,std=False):
    X=np.atleast_2d(X); out=[gp.predict((tr(X[i:i+5000])-mu)/sd,return_std=std) for i in range(0,len(X),5000)]
    return (np.concatenate([o[0] for o in out]),np.concatenate([o[1] for o in out])) if std else np.concatenate(out)
rng=np.random.default_rng(0)
lo=np.array([2.5,1.0,4,-2,0,11,10]); hi=np.array([14,3.5,10,8,5,20,30])
R=lo+(hi-lo)*rng.random((40000,7)); R[:,[2,4,5]]=np.round(R[:,[2,4,5]])
pr=pred(R); i=np.argsort(-pr)[:8]
for j in i: print(np.round(R[j],2), round(pr[j],1))
pk=R[i[0]]
for k in range(7):
    xs=np.linspace(lo[k],hi[k],11); A=np.tile(pk,(11,1)); A[:,k]=xs
    if k in (2,4,5): A[:,k]=np.round(xs)
    m,s=pred(A,True)
    print(K[k], ' '.join(f'{a:.1f}:{b:.1f}({c:.1f})' for a,b,c in zip(A[:,k],m,s)))
print('grid mid_fine x curvature (others at peak)')
cs=[1.4,1.7,2.0,2.3,2.6,2.9,3.2]
print('      '+' '.join(f'{c:6.1f}' for c in cs))
for mf in [3,4,5,6,7,8,9,10,12]:
    A=np.tile(pk,(len(cs),1)); A[:,0]=mf; A[:,1]=cs
    print(f'{mf:5.1f} '+' '.join(f'{v:6.1f}' for v in pred(A)))
print('data coverage: mid_fine quantiles', np.round(np.quantile(X[:,0],[0,.1,.5,.9,1]),1), 'curv', np.round(np.quantile(X[:,1],[0,.1,.5,.9,1]),2))
