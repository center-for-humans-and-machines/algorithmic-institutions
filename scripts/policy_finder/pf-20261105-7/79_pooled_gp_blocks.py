"""Pooled GP of v3-v6 sweeps (independent seeds) + finals with their seed-block effects removed.
Coordinates as v6 (mid_fine, curvature, window, gate, floor, end, net_fine at c=0; gate 8 = off).
Block effect b_k = mean over block k of (final - GP(sweep only) prediction); finals then enter with
noise se_paired (~0.6) after subtracting b_k. Prints block effects, peak, profiles, predictions at finals."""
import json, sys, numpy as np, pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
VERS=['pf-20261105-7-v3','pf-20261105-7-v4','pf-20261105-7-v5','pf-20261105-7-v6']
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
    for p in json.load(open(f'plots/simulation/policy_finder/{v}_sweep/sweep.json'))['points']:
        X.append(conv(p['params']));Y.append(p['pool']);S.append(p['pool_se'])
X=np.array(X,float);Y=np.array(Y);S=np.array(S)
F=[];FY=[];FB=[];FN=[]
for v in VERS:
    for k,p in enumerate(json.load(open(f'plots/simulation/policy_finder/{v}_sweep/final/sweep.json'))['points']):
        F.append(conv(p['params']));FY.append(p['pool']);FB.append(k);FN.append(p['name'].replace('pf-20261105-7-',''))
F=np.array(F,float);FY=np.array(FY);FB=np.array(FB)
def tr(X): Z=np.array(X,float).copy(); Z[:,0]=np.log(Z[:,0]); Z[:,1]=np.log(Z[:,1]); return Z
Z=tr(X); mu=Z.mean(0); sd=Z.std(0)
def fit(Xa,Ya,Sa):
    gp=GaussianProcessRegressor(Ck(1.)*Matern(length_scale=np.ones(7),nu=2.5)+WhiteKernel(0.01),alpha=(Sa/Ya.std())**2,normalize_y=True,n_restarts_optimizer=2,random_state=0)
    gp.fit((tr(Xa)-mu)/sd,Ya); return gp
gp=fit(X,Y,S)
SEP=float(sys.argv[1]) if len(sys.argv)>1 else 0.6
for it in range(3):
    pf=gp.predict((tr(F)-mu)/sd)
    b=np.array([np.mean((FY-pf)[FB==k]) for k in range(6)])
    b-=0  # absolute level: block effects are relative to the sweep-based surface
    gp=fit(np.vstack([X,F]),np.concatenate([Y,FY-b[FB]]),np.concatenate([S,np.full(len(F),SEP)]))
    print('iter',it,'block effects',np.round(b,2))
print(gp.kernel_)
pred=lambda A: gp.predict((tr(np.atleast_2d(A))-mu)/sd,return_std=True)
m,s=pred(F)
print('finals: name  real  block-adjusted  fit')
for n,y,bb,mm in sorted(zip(FN,FY,FB,m),key=lambda r:-(r[1]-b[r[2]])): print(f'  {n:10s} {y:7.2f} {y-b[bb]:7.2f} {mm:7.2f}')
rng=np.random.default_rng(0)
lo=np.array([3.5,1.4,4,-2,0,11,10]); hi=np.array([12,3.4,10,8,5,20,30])
R=lo+(hi-lo)*rng.random((60000,7)); R[:,[2,4,5]]=np.round(R[:,[2,4,5]])
pr=gp.predict((tr(R)-mu)/sd); i=np.argsort(-pr)[:10]
for j in i: print(np.round(R[j],2), round(pr[j],2))
pk=R[i[0]]
for k in range(7):
    xs=np.linspace(lo[k],hi[k],11); A=np.tile(pk,(11,1)); A[:,k]=xs
    if k in (2,4,5): A[:,k]=np.round(xs)
    mm,ss=pred(A)
    print(K[k], ' '.join(f'{a:.1f}:{b_:.1f}({c:.1f})' for a,b_,c in zip(A[:,k],mm,ss)))
np.save('/tmp/pf7_blocks.npy',b)
