# Within-block regression over the 36 finals: what does a point of fine (per player) in each phase
# buy in later contribution? payoff value per point = 0.6*dC/dp - 2 (p counted per player per round).
import pandas as pd, numpy as np, json
base='plots/simulation/policy_finder/'
vs=['pf-20261105-7','pf-20261105-7-v2','pf-20261105-7-v3','pf-20261105-7-v4','pf-20261105-7-v5','pf-20261105-7-v6']
rows=[]
for v in vs:
    d=pd.read_parquet(base+v+'_sweep/final/per_round.parquet'); d=d[d.contribution_valid]
    for k,r in enumerate(d.run.unique()):
        x=d[d.run==r]; c=x.groupby('round_number').contribution.mean(); p=x.groupby('round_number').punishment.mean()
        rows.append(dict(block=k,C=c.sum(),C1_6=c[1:7].sum(),C7_15=c[7:16].sum(),C16=c[16:].sum(),p0=p[0],p1_2=p[1:3].sum(),p3_6=p[3:7].sum(),p7_15=p[7:16].sum(),p16=p[16:].sum()))
T=pd.DataFrame(rows)
B=pd.get_dummies(T.block).values.astype(float)
for feats in (['p0','p1_2','p3_6','p7_15'],['p0','p3_6','p7_15'],['p1_2','p3_6','p7_15']):
  Xf=np.hstack([B,T[feats].values])
  for y in ['C','C16']:
    coef,res,rk,sv=np.linalg.lstsq(Xf,T[y].values,rcond=None)
    r=T[y].values-Xf@coef; s2=(r**2).sum()/(len(r)-Xf.shape[1]); cov=s2*np.linalg.inv(Xf.T@Xf)
    se=np.sqrt(np.diag(cov))[-len(feats):]
    print(y, ' '.join(f'{f}:{a:.2f}({b:.2f})' for f,a,b in zip(feats,coef[-len(feats):],se)), f'resid sd {np.sqrt(s2):.2f}')
print(T[['p0','p1_2','p3_6','p7_15','p16']].describe().round(2))
