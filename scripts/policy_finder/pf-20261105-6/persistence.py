"""Humanlike sim: effect of a punishment at round t on own contribution at t+k, matched on own c_t and others' mean, low contributors.
Also episode-level: early punishment (rounds 0-3) vs late contributions, controlling for round-0 contributions."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
s=pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
s=s[s.run.str.contains('humanlike')].sort_values(['episode','participant_code','round_number'])
s.loc[~s.contribution_valid,'contribution']=np.nan
W=s.pivot_table(index=['episode','participant_code'],columns='round_number',values='contribution')
P=s.pivot_table(index=['episode','participant_code'],columns='round_number',values='punishment')
C=W.values; Pm=P.values
ep=W.index.get_level_values(0).values
# others mean
df=pd.DataFrame(C,index=ep); om=df.groupby(level=0).transform('sum').values
cnt=pd.DataFrame(~np.isnan(C),index=ep).groupby(level=0).transform('sum').values
oth=(om-np.nan_to_num(C))/(cnt-1)
res={}
for k in [1,2,4,8]:
    rows=[]
    for t in range(2,23-k):
        rows.append(pd.DataFrame(dict(c=C[:,t],o=oth[:,t],p=Pm[:,t],pprev=Pm[:,t-1],ck=C[:,t+k])))
    d=pd.concat(rows).dropna()
    d=d[d.c<=9]
    d['cb']=d.c.round(-1)//5; d['ob']=(d.o//5)
    d['pb']=pd.cut(d.p,[-1,0,3,8,30])
    d['dk']=d.ck-d.c
    # within cell differences vs p=0
    tab=d.groupby(['cb','ob','pb']).dk.mean().unstack()
    base=tab.iloc[:,0]
    diff=tab.sub(base,axis=0); w=d.groupby(['cb','ob']).size()
    res[k]=(diff.mul(w,axis=0).sum()/w[diff.notna().all(axis=1)].sum()).round(2)
print('Effect on c_{t+k}-c_t vs unpunished, low contributors (c<=9), matched on c and others:'); print(pd.DataFrame(res))
# episode level
g=pd.DataFrame(C,index=ep).groupby(level=0).mean(); gp=pd.DataFrame(Pm,index=ep).groupby(level=0).mean()
E=pd.DataFrame(dict(c0=g[0], pe=gp[[0,1,2,3]].mean(1), pl=gp[list(range(4,24))].mean(1), cl=g[list(range(12,24))].mean(1)))
import numpy.linalg as la
X=np.c_[np.ones(len(E)),E.c0,E.pe,E.pl]; b=la.lstsq(X,E.cl,rcond=None)[0]
print('late contrib ~ 1 + c0 + early pun + mid/late pun:', b.round(3))

# spillover: others' change when someone else is punished (sum of punishment on others at t), matched on own c and others mean
rows=[]
Ps=pd.DataFrame(Pm,index=ep).groupby(level=0).transform('sum').values
for t in range(2,19):
    rows.append(pd.DataFrame(dict(c=C[:,t],o=oth[:,t],p=Pm[:,t],po=Ps[:,t]-Pm[:,t],c4=C[:,t+4],c1=C[:,t+1])))
d=pd.concat(rows).dropna(); d=d[d.p==0]
X=np.c_[np.ones(len(d)),d.c,d.o,d.po]
for col in ['c1','c4']:
    print('spillover',col,'~1+c+others+pun_on_others:',la.lstsq(X,d[col]-d.c,rcond=None)[0].round(3))
