"""Play candidate rules against the surrogate of 06 (no timeouts). Rough sizing only."""
import numpy as np, pickle
m,R0=pickle.load(open("/tmp/surrogate.pkl","rb"))
rng=np.random.default_rng(0)
def play(rule, n=1500):
    C=np.zeros((n,24,4));P=np.zeros((n,24,4))
    C[:,0]=R0[rng.integers(len(R0),size=n)]
    for t in range(24):
        P[:,t]=np.clip(np.floor(rule(C[:,:t+1],P[:,:t],t)),0,30)
        if t==23: break
        tl=max(t-1,0); feats=[]
        for i in range(4):
            o=[j for j in range(4) if j!=i]
            feats.append(np.stack([C[:,t,i],C[:,tl,i],C[:,t,o].mean(1),C[:,tl,o].mean(1),P[:,t,i],P[:,tl,i] if t>0 else 0*P[:,t,i],P[:,t,o].sum(1),np.full(n,t),C[:,:t+1,i].mean(1)],1))
        pr=m.predict_proba(np.concatenate(feats))  # [4n,21]
        u=rng.random((len(pr),1)); y=(pr.cumsum(1)<u).sum(1).clip(0,20)
        C[:,t+1]=m.classes_[y].reshape(4,n).T
    pay=(20+0.6*C.mean(2)-2*P.mean(2)).sum(1)
    return pay.mean(), pay.std()/np.sqrt(n), C.mean((0,2))[[0,5,11,17,23]].round(1), P.mean()
def deficit(thr,s,T,base=0):
    return lambda C,P,t: ((C[:,-1]<thr)*(base+s*(thr-C[:,-1])))*(t<T)
print("zero",play(lambda C,P,t:0*C[:,-1]))
for thr in [10,15,20]:
  for s in [0.5,1,2]:
    for T in [6,12,23]:
      print(thr,s,T,play(deficit(thr,s,T)))
