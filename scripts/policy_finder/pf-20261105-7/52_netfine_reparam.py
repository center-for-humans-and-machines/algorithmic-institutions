"""v3 real sweep: quadratic regression of the score on the params with the net fine expressed as an
absolute amount at c=1 (min(30, scale*F*(0.95)^curv)); profile of the fitted score along it."""
import numpy as np, json
pts=json.load(open('plots/simulation/policy_finder/pf-20261105-7-v3_sweep/sweep.json'))['points']
y=np.array([p['pool'] for p in pts]); P=[p['params'] for p in pts]
nf=np.array([min(30,p['net_scale']*p['free_rider_fine']*0.95**p['curvature']) for p in P])
X0=np.array([[p['free_rider_fine'],np.log(p['curvature']),p['window'],p['net_floor'],p['net_end']] for p in P])
X0=np.column_stack([X0,nf]); names=['F','logcurv','window','floor','end','netfine']
mu=X0.mean(0); sd=X0.std(0); Z=(X0-mu)/sd
cols=[np.ones(len(y))]+[Z[:,i] for i in range(6)]+[Z[:,i]**2 for i in range(6)]+[Z[:,i]*Z[:,j] for i in range(6) for j in range(i+1,6)]
A=np.column_stack(cols); b,*_=np.linalg.lstsq(A,y,rcond=None)
res=y-A@b; print('R2',1-res.var()/y.var(), 'resid sd',res.std())
def pred(z):
    c=[1]+list(z)+[v**2 for v in z]+[z[i]*z[j] for i in range(6) for j in range(i+1,6)]
    return np.dot(b,c)
# peak by grid search over z in data range
best=None
rng=np.random.default_rng(0)
for _ in range(200000):
    x=np.array([rng.uniform(X0[:,i].min(),X0[:,i].max()) for i in range(6)]); x[2:5]=np.round(x[2:5])
    v=pred((x-mu)/sd)
    if best is None or v>best[0]: best=(v,x)
print('peak',round(best[0],1),dict(zip(names,best[1].round(2))))
x=best[1].copy()
for v in [8,12,16,20,24,28,30]:
    x[5]=v; print(' netfine',v,round(pred((x-mu)/sd),1))
print('netfine distribution',np.percentile(nf,[10,50,90]).round(1), 'share at 30',(nf>=30).mean())
