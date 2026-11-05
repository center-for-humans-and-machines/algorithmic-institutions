# Per-episode punishment slope on deficit in humanlike sims vs outcomes
import pandas as pd, numpy as np
pd.set_option('display.width',200)
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d=d[d.run.str.contains('humanlike')&d.contribution_valid].copy()
d['def']=20-d.contribution
d['early']=d.round_number<8
rows=[]
for e,z in d.groupby('episode'):
    ze=z[z.early]
    X=np.c_[np.ones(len(ze)),ze['def']]
    b=np.linalg.lstsq(X,ze.punishment,rcond=None)[0]
    rows.append(dict(slope=b[1],icpt=b[0],c0=z[z.round_number==0].contribution.mean(),
        p_early=ze.punishment.mean(),c_late=z[z.round_number>=12].contribution.mean(),
        p_late=z[z.round_number>=12].punishment.mean(),pay=z.groupby('round_number').payoff.mean().sum()))
s=pd.DataFrame(rows)
s['sb']=pd.qcut(s.slope,5)
print(s.groupby('sb',observed=True)[['slope','icpt','c0','p_early','c_late','p_late','pay']].mean().round(2))
X=np.c_[np.ones(len(s)),s.slope,s.c0,s.icpt]
print('c_late ~ 1,slope,c0,icpt', np.linalg.lstsq(X,s.c_late,rcond=None)[0].round(2))
print('pay ~ 1,slope,c0,icpt', np.linalg.lstsq(X,s.pay,rcond=None)[0].round(2))
z=pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
z=z[z.run.str.contains('zero')&z.contribution_valid]
zz=pd.DataFrame(dict(c0=z[z.round_number==0].groupby('episode').contribution.mean(), c_late=z[z.round_number>=12].groupby('episode').contribution.mean(), pay=z.groupby(['episode','round_number']).payoff.mean().groupby('episode').sum()))
zz['cb']=pd.qcut(zz.c0,5); s['cb']=pd.qcut(s.c0,5)
print('zero by c0', zz.groupby('cb',observed=True)[['c_late','pay']].mean().round(2))
print('humanlike by c0', s.groupby('cb',observed=True)[['c_late','pay','p_early']].mean().round(2))
