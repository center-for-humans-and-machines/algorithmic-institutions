# v4: per design point of v3's sweep, the paired effect on the score (sum of paired lead changes over its
# punished games / its 500 games), and how it depends on the parameters.
import pandas as pd, numpy as np, json
pd.set_option('display.width',220)
df=pd.read_parquet('/tmp/v4_paired.parquet'); v3=df[df.v=='v3']
sw=json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261112-1-v3_sweep/sweep.json'))
P=pd.DataFrame([dict(point=p['name'],score=p['pool'],**p['params']) for p in sw['points']])
g=v3.groupby('point').agg(s=('dlead','sum'),ss=('dlead',lambda x:(x**2).sum()),n=('dlead','size'))
P=P.merge(g,left_on='point',right_index=True,how='left').fillna({'s':0,'ss':0,'n':0})
P['eff']=P.s/500; P['eff_se']=np.sqrt(P.ss/500-P.eff**2)/np.sqrt(500); P['touched']=P.n/500
P=P[P.n_full<9]
for k in ['n_full','bar','amount','n_push','last_dec']:
    q=P.copy()
    if k=='amount': q['amount']=pd.cut(q.amount,[4,10,15,20,25,30])
    print(q.groupby(k,observed=True).agg(npts=('eff','size'),eff=('eff','mean'),se=('eff','sem'),touched=('touched','mean')).round(3))
import numpy.linalg as la
X=np.column_stack([np.ones(len(P)),P.n_full==8,P.bar,P.amount,P.n_push,P.last_dec,P.amount**2/30])
b=la.lstsq(X,P.eff,rcond=None)[0]; print('ols const,nf8,bar,amount,n_push,last_dec,amount^2/30',b.round(4))
print(P.sort_values('eff',ascending=False).head(15).round(3))
print('mean eff of punishing points',P.eff.mean(), 'sd',P.eff.std())
for nf in [7,8]:
  q=P[(P.n_full==nf)&(P.bar>=3)&(P.amount>=14)&(P.last_dec<=3)]
  print('region n_full',nf,'bar>=3 amount>=14 last_dec<=3: points',len(q),'eff %.3f se %.3f touched %.3f'%(q.eff.mean(),q.eff.sem(),q.touched.mean()))
q=P[(P.bar>=3)&(P.amount>=14)&(P.last_dec<=3)]
print('both', len(q), q.eff.mean(), q.eff.sem())
print(q.groupby('n_push').eff.agg(['mean','sem','size']))
