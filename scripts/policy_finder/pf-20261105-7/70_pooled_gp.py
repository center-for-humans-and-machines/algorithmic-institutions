"""Pool the real sweeps of v3, v4, v5 (and their finals) in v5's parameter space and fit a GP.
v3's net fine at c=0 is min(30, F*net_scale); v3/v4 have no gate (gate_margin := 8, which is
practically off). Prints the fitted peak, profiles, and the predicted score of each final point."""
import json, sys, numpy as np
VERS=sys.argv[1:] or ['pf-20261105-7-v3','pf-20261105-7-v4','pf-20261105-7-v5']
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
K=['free_rider_fine','curvature','window','gate_margin','net_floor','net_end','net_fine']
def conv(p):
    q=dict(p)
    if 'net_scale' in q: q['net_fine']=min(30.,q['free_rider_fine']*q.pop('net_scale'))
    q.setdefault('gate_margin',8.)
    q['gate_margin']=min(q['gate_margin'],8.)
    if q['net_end']<=q['window']: q['net_end']=q['window']; q['net_fine']=0.  # no net
    return [q[k] for k in K]
X=[];Y=[];S=[];src=[]
for v in VERS:
    for f in ['sweep','final/sweep']:
        d=json.load(open(f'plots/simulation/policy_finder/{v}_sweep/{f}.json'))
        for p in d['points']:
            X.append(conv(p['params']));Y.append(p['pool']);S.append(p['pool_se']);src.append(v+'/'+f)
X=np.array(X,float);Y=np.array(Y);S=np.array(S)
def tr(X):
    Z=X.copy(); Z[:,1]=np.log(Z[:,1]); return Z
Z=tr(X); mu=Z.mean(0); sd=Z.std(0); Zn=(Z-mu)/sd
gp=GaussianProcessRegressor(Ck(1.)*Matern(length_scale=np.ones(7),nu=2.5)+WhiteKernel(0.01),alpha=(S/Y.std())**2,normalize_y=True,n_restarts_optimizer=3,random_state=0)
gp.fit(Zn,Y)
print(gp.kernel_)
pred=lambda X: np.concatenate([gp.predict((tr(np.atleast_2d(X)[i:i+5000])-mu)/sd) for i in range(0,len(np.atleast_2d(X)),5000)])
# random search for the peak inside a box
rng=np.random.default_rng(0)
lo=np.array([22,1.6,4,-2,0,11,10]); hi=np.array([50,3.0,10,8,5,20,30])
R=lo+(hi-lo)*rng.random((40000,7)); R[:,[2,4,5]]=np.round(R[:,[2,4,5]])
pr=pred(R); i=np.argsort(-pr)[:10]
for j in i: print(np.round(R[j],2), round(pr[j],1))
pk=R[i[0]]
print('profiles at peak')
for k in range(7):
    xs=np.linspace(lo[k],hi[k],9); A=np.tile(pk,(9,1)); A[:,k]=xs
    if k in (2,4,5): A[:,k]=np.round(xs)
    print(K[k], ' '.join(f'{a:.1f}:{b:.1f}' for a,b in zip(A[:,k],pred(A))))
print('finals: real vs fitted')
for x,y,s in zip(X,Y,src):
    if 'final' in s: print(s.split('/')[0][-3:], np.round(x,2), round(y,1), round(pred(x)[0],1))
