import numpy as np, torch as th, pandas as pd, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur2 import feats, Net
th.manual_seed(0); th.set_num_threads(2)
a=np.load('/tmp/pf7_arr.npz'); C=th.tensor(a['C']); P=th.tensor(a['P'])
# add zero reference games
ref=pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet',columns=['run','episode','participant_code','round_number','contribution','punishment'])
ref=ref.sort_values(['run','episode','participant_code','round_number'])
Cr=th.tensor(ref.contribution.values.reshape(-1,4,24),dtype=th.float32); Pr=th.tensor(ref.punishment.values.reshape(-1,4,24),dtype=th.float32)
C=th.cat([C,Cr,Cr,Cr]); P=th.cat([P,Pr,Pr,Pr])
B=C.shape[0]; perm=th.randperm(B); C=C[perm]; P=P[perm]
nval=4000
X=[];Y=[]
for t in range(23):
    X.append(feats(C,P,t)); Y.append(C[...,t+1].long())
X=th.stack(X,2).reshape(B,-1,20); Y=th.stack(Y,2).reshape(B,-1)
Xtr=X[nval:].reshape(-1,20); Ytr=Y[nval:].reshape(-1); Xv=X[:nval].reshape(-1,20); Yv=Y[:nval].reshape(-1)
print(Xtr.shape)
net=Net(); opt=th.optim.Adam(net.parameters(),1e-3)
lf=nn=th.nn.CrossEntropyLoss()
bs=8192; N=Xtr.shape[0]
for ep in range(3):
    idx=th.randperm(N)
    for i in range(0,N,bs):
        j=idx[i:i+bs]
        loss=lf(net(Xtr[j]),Ytr[j]); opt.zero_grad(); loss.backward(); opt.step()
    for gr in opt.param_groups: gr['lr']*=0.4
    with th.no_grad(): print(ep, loss.item(), lf(net(Xv),Yv).item(), flush=True)
th.save(net.state_dict(),'scripts/policy_finder/pf-20261105-7/sur2.pt')
