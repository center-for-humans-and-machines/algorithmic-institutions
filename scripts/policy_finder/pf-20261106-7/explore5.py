# how did the mid-punishment managers punish? per-manager-style punishment schedule
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-7')
from load import load
import numpy as np, pandas as pd
df = load()
g = df.groupby(['game','round_number','group_id']).agg(gc=('contribution','mean'), n=('player_id','size'),pool=('common_good','first')).reset_index()
df = df.merge(g, on=['game','round_number','group_id'])
df['gap']=df.gc-df.contribution
s = df.groupby(['game','group_id']).punishment.mean().rename('mp').reset_index()
s['style']=pd.cut(s.mp,[-1,0.53,1.14,1.66,2.76,30],labels=['none','low','mid','high','vhigh'])
df=df.merge(s,on=['game','group_id'])
print(df.pivot_table(index=pd.cut(df.contribution,[-1,0,4,9,14,20]),columns='style',values='punishment',observed=True).round(2))
print(df.pivot_table(index=pd.cut(df.round_number,[-1,3,7,11,15,19,23]),columns='style',values='punishment',observed=True).round(2))
print(df.pivot_table(index=pd.cut(df.round_number,[-1,3,7,11,15,19,23]),columns='style',values='contribution',observed=True).round(2))
print(df.pivot_table(index=pd.cut(df.round_number,[-1,3,7,11,15,19,23]),columns='style',values='n',observed=True).round(2))
print(s.style.value_counts())
