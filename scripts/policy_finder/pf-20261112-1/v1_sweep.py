# v2: how did v1's design points score, against how much they punished and when?
import json, pandas as pd, numpy as np, glob
import statsmodels.formula.api as smf
base='plots/simulation/policy_finder/pf-20261112-1_sweep/'
s=json.load(open(base+'sweep.json'))
pts=pd.DataFrame([dict(name=p['name'],pool=p['pool'],se=p['pool_se'],mem=p['members'],own=p['own'],oth=p['other'],**p['params']) for p in s['points']])
print(pts.describe().round(2))
# punishment volume per point from per_round
rows=[]
for f in sorted(glob.glob(base+'per_round_p*.parquet')):
    import pyarrow.parquet as pq
    d=pq.read_table(f,columns=['round_number','punishment','contribution','agent_group','run'],read_dictionary=['run']).to_pandas()
    cats=pd.Series(d.run.cat.categories)
    pair=cats.str.split('by ').str[1]
    seat0=~pair.str.startswith('ah_vs')
    pt=np.where(seat0,pair.str.split('_vs_').str[0],pair.str.split('_vs_').str[1])
    codes=d.run.cat.codes.values
    d['pt']=pd.Categorical.from_codes(codes,categories=cats).rename_categories(dict(zip(cats,cats)))
    d['pt']=pd.Series(pt)[codes].values
    d['mine']=np.where(seat0.values[codes],d.agent_group==0,d.agent_group==1)
    d=d.drop(columns='run')
    m=d[d.mine].copy()
    m['early']=m.round_number<4
    g=m.groupby('pt').agg(pun=('punishment','mean'),con=('contribution','mean'))
    ge=m[m.early].groupby('pt').punishment.mean().rename('pun_early')
    gl=m[~m.early].groupby('pt').punishment.mean().rename('pun_late')
    rows.append(pd.concat([g,ge,gl],axis=1))
v=pd.concat(rows); v.index.name='name'
pts=pts.merge(v.reset_index(),on='name')
pts.to_csv('/tmp/v1pts.csv',index=False)
print(pts.sort_values('pool',ascending=False).head(20).round(2).to_string())
print(pts[['pool','mem','pun','pun_early','pun_late','con']].corr().round(2))
pts['pb']=pd.cut(pts.pun,[-.01,0.05,0.3,0.7,1.5,3,30])
print(pts.groupby('pb',observed=True)[['pool','mem','con','pun_early']].agg(['mean','size']).round(2))
print(smf.ols('pool ~ pun_early + pun_late',pts).fit().params.round(2))
