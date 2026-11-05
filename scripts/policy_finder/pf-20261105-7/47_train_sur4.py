"""Train the sur3 architecture on the real games of all three sweeps + finals (v1, v2, v3).
argv[1]: 'all3' -> sur3_all3.pt ; 'v12' -> same data as sur3_all (for a held-out check on v3)."""
import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur3 import feats_all, Net, NF
th.manual_seed(0); th.set_num_threads(2)
which=sys.argv[1]
vs=['pf-20261105-7','pf-20261105-7-v2']+(['pf-20261105-7-v3'] if which=='all3' else [])
files=[f'{v}_{t}' for v in vs for t in ('sweep','final')]
C=np.concatenate([np.load(f'/tmp/pf7/{f}.npz')['C'] for f in files]); P=np.concatenate([np.load(f'/tmp/pf7/{f}.npz')['P'] for f in files])
C=th.tensor(C); P=th.tensor(P); B=C.shape[0]; perm=th.randperm(B); C=C[perm]; P=P[perm]
nval=3000; bs=512
net=Net(); opt=th.optim.Adam(net.parameters(),1e-3); lf=th.nn.CrossEntropyLoss()
def batch(i,j):
    c=C[i:j].float(); p=P[i:j].float()
    X=feats_all(c,p)[:,:,:-1].reshape(-1,NF); Y=c[...,1:].long().reshape(-1)
    return X,Y
Xv,Yv=batch(0,nval)
epochs=int(sys.argv[2]) if len(sys.argv)>2 else 2
for ep in range(epochs):
    for k,i in enumerate(range(nval,B,bs)):
        X,Y=batch(i,min(i+bs,B))
        loss=lf(net(X),Y); opt.zero_grad(); loss.backward(); opt.step()
        if k%200==0:
            with th.no_grad(): print(ep,k,round(loss.item(),4),round(lf(net(Xv),Yv).item(),4),flush=True)
    for gr in opt.param_groups: gr['lr']*=0.3
th.save(net.state_dict(),f'scripts/policy_finder/pf-20261105-7/sur3_{which}.pt')
