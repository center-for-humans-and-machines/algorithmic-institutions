# contribution response to punishment
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-7')
from load import load
import numpy as np, pandas as pd
df = load()
g = df.groupby(['game','round_number','group_id']).agg(gc=('contribution','mean'), n=('player_id','size')).reset_index()
df = df.merge(g, on=['game','round_number','group_id'])
df['others_c']=(df.gc*df.n-df.contribution)/(df.n-1).clip(lower=1)
d=df.copy()
d['c_next']=d.groupby(['game','player_id']).contribution.shift(-1)
d['g_next']=d.groupby(['game','player_id']).group_id.shift(-1)
d=d[(d.round_number<23)&(d.g_next==d.group_id)&(d.player_no_input==0)]
d['dc']=d.c_next-d.contribution
d['pb']=pd.cut(d.punishment,[-1,0,2,5,10,30])
d['cb']=pd.cut(d.contribution,[-1,4,9,14,20])
print(d.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean').round(2))
print(d.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size'))
import statsmodels.formula.api as smf
d['gap']=d.others_c-d.contribution
m=smf.ols('dc ~ punishment + I(punishment**2)+ contribution + gap', data=d).fit(); print(m.params.round(3), m.bse.round(3))
# human manager: punishment vs contribution and gap
print(df.groupby(pd.cut(df.contribution,[-1,0,4,9,14,19,20])).punishment.agg(['mean','size']).round(2))
df['gap']=df.others_c-df.contribution
print(df.groupby(pd.cut(df.gap,[-21,-5,-1,1,5,10,21])).punishment.mean().round(2))
