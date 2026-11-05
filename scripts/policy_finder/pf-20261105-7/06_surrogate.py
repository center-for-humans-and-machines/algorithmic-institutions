"""Crude surrogate of the exp1 clones: a gradient-boosted classifier of c(t+1) (21 levels)
from c, lagged c, others' mean, own and others' punishment (now and lagged), round,
fitted on both reference sims. Used only to rough-size the rule's ranges; its
punishment effect is confounded and must not be trusted for the values themselves."""
import pandas as pd, numpy as np, pickle, sys
from sklearn.ensemble import HistGradientBoostingClassifier
s = pd.read_parquet("plots/simulation/29_EXP1_reference/per_round.parquet")
s.loc[~s.contribution_valid,"contribution"]=0
s["pid"]=s.participant_code.str.split("_").str[0].astype(int)
X=[];Y=[];R0=[]
for run, r in s.groupby("run"):
    C=r.pivot_table(index=["episode","round_number"],columns="pid",values="contribution").values.reshape(-1,24,4)
    P=r.pivot_table(index=["episode","round_number"],columns="pid",values="punishment").values.reshape(-1,24,4)
    R0.append(C[:,0,:])
    for t in range(23):
        tl=max(t-1,0)
        for i in range(4):
            o=[j for j in range(4) if j!=i]
            f=np.stack([C[:,t,i],C[:,tl,i],C[:,t,o].mean(1),C[:,tl,o].mean(1),P[:,t,i],P[:,tl,i] if t>0 else 0*P[:,t,i],P[:,t,o].sum(1),np.full(len(C),t), C[:,:t+1,i].mean(1)],1)
            X.append(f);Y.append(C[:,t+1,i])
X=np.concatenate(X);Y=np.concatenate(Y).astype(int)
m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.1,max_leaf_nodes=31).fit(X,Y)
pickle.dump((m,np.concatenate(R0)),open("/tmp/surrogate.pkl","wb"))
print("fitted", X.shape)
