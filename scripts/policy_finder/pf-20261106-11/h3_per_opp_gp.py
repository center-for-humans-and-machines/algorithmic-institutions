# v5: v3+v4 sweeps share the code; fit a GP per opponent over (log rate, delta, decay, mid) on 512 points, find per-opponent optima and the loss of the common optimum
import json, numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
P=[]
for v in ['pf-20261106-11-v3','pf-20261106-11-v4']:
    P+=json.load(open(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/{v}_sweep/sweep.json'))['points']
X=np.array([[np.log(p['params']['rate']),p['params']['delta'],p['params']['decay'],p['params']['mid']] for p in P])
lo=np.array([np.log(2.5),-3.5,0.0,0.2]); hi=np.array([np.log(20),1.0,0.8,1.6])
Xs=(X-lo)/(hi-lo)
rng=np.random.default_rng(0); G=rng.uniform(size=(40000,4)); Gr=G*(hi-lo)+lo
preds={}
for o in ['opponent_1','opponent_2','opponent_3','opponent_4','all']:
    y=np.array([p['pool'] if o=='all' else p['pools'][o]['pool'] for p in P])
    gp=GaussianProcessRegressor(C(10)*RBF([0.5]*4,(0.05,10))+WhiteKernel(4),normalize_y=True).fit(Xs,y)
    mu=gp.predict(G); preds[o]=mu; i=mu.argmax()
    print(o, 'ls',np.round(gp.kernel_.k1.k2.length_scale,2), 'peak',round(mu[i],1),'at rate %.1f delta %.2f decay %.2f mid %.2f'%(np.exp(Gr[i,0]),*Gr[i,1:]))
avg=np.mean([preds[f'opponent_{k}'] for k in range(1,5)],0); i=avg.argmax()
print('common optimum avg',round(avg[i],1),'per-opp best avg',round(np.mean([preds[f'opponent_{k}'].max() for k in range(1,5)]),1))
for k in range(1,5): print(k,'at common',round(preds[f'opponent_{k}'][i],1),'own best',round(preds[f'opponent_{k}'].max(),1))
# slices along mid and decay at the common optimum, per opponent
base=G[i].copy()
for j,name in [(3,'mid'),(2,'decay'),(0,'lograte'),(1,'delta')]:
    xs=np.linspace(0,1,6); Z=np.repeat(base[None],6,0); Z[:,j]=xs
    print(name, np.round(xs*(hi[j]-lo[j])+lo[j],2))
    for k in range(1,5):
        y=np.array([p['pools'][f'opponent_{k}']['pool'] for p in P])
        gp=GaussianProcessRegressor(C(10)*RBF([0.5]*4,(0.05,10))+WhiteKernel(4),normalize_y=True).fit(Xs,y)
        mu,sd=gp.predict(Z,return_std=True); print('  opp',k,np.round(mu,1),'sd',np.round(sd.mean(),1))
