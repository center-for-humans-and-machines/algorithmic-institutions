import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from load import load
import pandas as pd, numpy as np
df=load()
g=df.groupby(['game','player_id'])
df['c1']=g.contribution.shift(-1); df['g1']=g.group_id.shift(-1)
s=df.groupby(['game','round_number','group_id']).contribution.agg(['sum','size']).reset_index()
df=df.merge(s,on=['game','round_number','group_id'])
df['oth']=(df['sum']-df.contribution)/(df['size']-1).replace(0,np.nan)
d=df[(df.player_no_input==0)&(df.g1==df.group_id)].dropna(subset=['oth','c1']).copy()
d['dc']=d.c1-d.contribution
d['cb']=pd.cut(d.contribution,[-1,4,8,12,16,20]); d['ob']=pd.cut(d.oth,[-1,6,10,14,21])
d['pun']=(d.punishment>=3).astype(int)
t=d.pivot_table(index=['cb','ob'],columns='pun',values='dc',aggfunc=['mean','size'],observed=True).round(2)
print(t.to_string())
