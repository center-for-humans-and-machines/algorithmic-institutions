# v5: per design point of v4's sweep, paired effect on the score (sum of paired lead changes / games of the point)
import pandas as pd, numpy as np, json
pd.set_option('display.width',220)
df=pd.read_parquet('/tmp/v5_paired.parquet'); cnt=pd.read_parquet('/tmp/v5_counts.parquet')
v=df[df.v=='v4']
c=cnt[cnt.v=='v4'].set_index('point').games
g=v.groupby('point').agg(s=('dlead','sum'),ss=('dlead',lambda x:(x**2).sum()),n=('dlead','size'))
sw=json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261112-1-v4_sweep/sweep.json'))
P=pd.DataFrame([dict(point=p['name'],score=p['pool'],**p['params']) for p in sw['points']])
P=P.merge(g,left_on='point',right_index=True,how='left').fillna({'s':0,'ss':0,'n':0})
P['games']=P.point.map(c); P['eff']=P.s/P.games; P['touched']=P.n/P.games
P['eff_se']=np.sqrt((P.ss/P.games-P.eff**2)/P.games)
print('points',len(P),'mean eff',P.eff.mean(),'sem',P.eff.sem(),'median se per point',P.eff_se.median())
for k in ['bar','amount','n_push','last_dec']:
    q=P.copy()
    if k=='amount': q['amount']=pd.cut(q.amount,[13,16,18,20,22,24,26,28,30])
    print(q.groupby(k,observed=True).agg(npts=('eff','size'),eff=('eff','mean'),se=('eff','sem'),touched=('touched','mean')).round(3))
print(pd.crosstab(P.bar,P.n_push,values=P.eff,aggfunc='mean').round(2))
print(pd.crosstab(pd.cut(P.amount,[13,18,23,30]),P.last_dec,values=P.eff,aggfunc='mean').round(2))
X=np.column_stack([np.ones(len(P)),P.bar,P.amount,P.n_push,P.last_dec==3,(P.amount-20)**2])
b,res,_,_=np.linalg.lstsq(X,P.eff,rcond=None); r=P.eff-X@b; s2=(r**2).sum()/(len(P)-X.shape[1])
se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
print('ols const,bar,amount,n_push,ld3,(amount-20)^2', b.round(4), se.round(4))
print('corr eff vs sweep score',np.corrcoef(P.eff,P.score)[0,1])
P.to_parquet('/tmp/v5_v4_points.parquet')
