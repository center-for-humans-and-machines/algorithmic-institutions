import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-6')
from load import load
import numpy as np, pandas as pd
df=load()
df['c_next']=df.groupby(['game','player_id']).contribution.shift(-1)
df['g_next']=df.groupby(['game','player_id']).group_id.shift(-1)
d=df[(df.g_next==df.group_id)&(df.player_no_input==0)].dropna(subset=['c_next']).copy()
d['dc']=d.c_next-d.contribution
d['cbin']=pd.cut(d.contribution,[-1,0,4,8,12,16,19,20])
d['pbin']=pd.cut(d.punishment,[-1,0,2,5,10,31])
print(d.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean',observed=True).round(2))
print(d.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='size',observed=True))
import statsmodels.formula.api as smf
d['pos']=(d.punishment>0).astype(int)
m=smf.ols('dc ~ contribution + punishment + I(punishment**2) + pos',data=d).fit(cov_type='cluster',cov_kwds={'groups':d.game}); print(m.summary2().tables[1].round(3))
# how did human managers punish as function of c and group mean
df['gmean']=df.groupby(['game','round_number','group_id']).contribution.transform('mean')
df['dev']=df.contribution-df.gmean
df['cb']=pd.cut(df.contribution,[-1,0,4,8,12,16,19,20])
print(df.groupby('cb',observed=True).punishment.agg(['mean',lambda x:(x>0).mean()]).round(2))
# longer horizon: contribution 2..4 rounds later
for k in [1,2,3]:
    df[f'c{k}']=df.groupby(['game','player_id']).contribution.shift(-k)
    df[f'g{k}']=df.groupby(['game','player_id']).group_id.shift(-k)
    e=df[(df[f'g{k}']==df.group_id)].dropna(subset=[f'c{k}'])
    e=e[e.contribution<=8]
    print(k, e.groupby(pd.cut(e.punishment,[-1,0,2,5,10,31]),observed=True).apply(lambda x:(x[f'c{k}']-x.contribution).mean()).round(2).to_dict())
