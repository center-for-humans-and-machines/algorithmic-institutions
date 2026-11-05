# Persistence and spillover of punishment in the humanlike reference sims
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d = d[d.run.str.contains('humanlike')].copy()
d['pl']=d.participant_code.str.split('_').str[0].astype(int)
d.loc[~d.contribution_valid,'contribution']=0
C = d.pivot_table(index=['episode','pl'],columns='round_number',values='contribution').values.reshape(-1,4,24)
P = d.pivot_table(index=['episode','pl'],columns='round_number',values='punishment').values.reshape(-1,4,24)
V = d.pivot_table(index=['episode','pl'],columns='round_number',values='contribution_valid').values.reshape(-1,4,24).astype(float)
E = C.shape[0]
rows=[]
for t in range(1,20):
    for i in range(4):
        o=[j for j in range(4) if j!=i]
        for k in (1,2,3,4):
            if t+k>23: continue
            rows.append(dict(t=t,k=k,c=C[:,i,t],p=P[:,i,t],po=P[:,o,t].sum(1),co=C[:,o,t].mean(1),
                             cprev=C[:,i,t-1],pprev=P[:,i,t-1],y=C[:,i,t+k],v=V[:,i,t]*V[:,i,t+k]))
for k in (1,2,3,4):
    rr=[r for r in rows if r['k']==k]
    cat=lambda n: np.concatenate([r[n] for r in rr])
    m=cat('v')>0
    c,p,po,co,y,cp,pp=[cat(n)[m] for n in ('c','p','po','co','y','cprev','pprev')]
    X=np.c_[np.ones(len(c)),c,p,po,co,cp,pp]
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    print(f'k={k}: c_(t+k) ~ 1 {b[0]:.2f}, c {b[1]:.3f}, p {b[2]:.3f}, p_others_sum {b[3]:.3f}, c_others_mean {b[4]:.3f}, c_prev {b[5]:.3f}, p_prev {b[6]:.3f}')
    # low contributors only
    lo=c<=5
    b=np.linalg.lstsq(X[lo],y[lo],rcond=None)[0]
    print(f'   c<=5: p {b[2]:.3f}, p_others_sum {b[3]:.3f}, c_others {b[4]:.3f}')
    hi=c>=15
    b=np.linalg.lstsq(X[hi],y[hi],rcond=None)[0]
    print(f'   c>=15: p {b[2]:.3f}, p_others_sum {b[3]:.3f}, c_others {b[4]:.3f}')
