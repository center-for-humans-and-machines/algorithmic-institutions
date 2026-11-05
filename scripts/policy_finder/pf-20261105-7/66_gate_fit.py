"""Quadratic fit of 64's random search of the gated family on the surrogate: top points, peak, profiles."""
import numpy as np
a=np.load('/tmp/pf7/gate_search.npy'); print(a.shape)
o=a[np.argsort(-a[:,-1])]
print('F gm w m floor E nf score')
for r in o[:15]: print(' '.join(f'{x:6.2f}' for x in r))
X=a[:,:7].copy(); X[:,1]=np.log(X[:,1]); mu=X.mean(0); sd=X.std(0)
def design(Z): return np.stack([np.ones(len(Z))]+[Z[:,i] for i in range(7)]+[Z[:,i]*Z[:,j] for i in range(7) for j in range(i,7)],1)
A=design((X-mu)/sd); b=np.linalg.lstsq(A,a[:,-1],rcond=None)[0]; pred=A@b
print('R2',round(1-((a[:,-1]-pred)**2).mean()/a[:,-1].var(),3),'resid sd',round((a[:,-1]-pred).std(),2))
Zs=np.random.default_rng(1).uniform(-1.7,1.7,(200000,7)); p=design(Zs)@b; k=p.argmax()
x=Zs[k]*sd+mu; x[1]=np.exp(x[1]); print('peak',np.round(x,2),round(p[k],1))
names=['F','gm','w','m','floor','E','nf']
for i in range(7):
    zz=np.tile(Zs[k],(9,1)); zz[:,i]=np.linspace(-1.7,1.7,9); pp=design(zz)@b
    xv=zz[:,i]*sd[i]+mu[i]; xv=np.exp(xv) if i==1 else xv
    print(f'{names[i]:5s}',' '.join(f'{u:.1f}:{v:.1f}' for u,v in zip(xv,pp)))
