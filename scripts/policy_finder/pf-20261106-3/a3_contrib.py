import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-3')
from load import load
import numpy as np, pandas as pd
df = load().sort_values(['game','player_id','round_number'])
g = df.groupby(['game','player_id'])
df['pc']=g.contribution.shift(1); df['pp']=g.punishment.shift(1); df['pg']=g.group_id.shift(1)
df['dc']=df.contribution-df.pc
d = df.dropna(subset=['pc'])
d = d[(d.player_no_input==0)]
stay = d[d.pg==d.group_id]
# change in contribution given prev punishment, by prev contribution bins
stay = stay.assign(cb=pd.cut(stay.pc,[-1,4,9,14,19,20]), pb=pd.cut(stay.pp,[-1,0,2,5,10,30]))
print(stay.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean').round(2))
print(stay.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size'))
import statsmodels.formula.api as smf
m = smf.ols('contribution ~ pc + pp + I(pp**2) + round_number', stay).fit()
print(m.params.round(3), m.bse.round(3))
# punishment of others in group (mean) effect
# effect of being punished on contribution two rounds later persistence
df['c2']=g.contribution.shift(-1)
d2=df.dropna(subset=['pc','c2']); d2=d2[(d2.player_no_input==0)]
m2 = smf.ols('c2 ~ pc + pp + contribution', d2).fit(); print('lead', m2.params.round(3))
# how did human managers punish: punishment vs contribution and vs group mean
df['gm'] = df.groupby(['game','round_number','group_id']).contribution.transform('mean')
df['dev']=df.contribution-df.gm
print(df.assign(cb=pd.cut(df.contribution,[-1,0,4,9,14,19,20])).groupby('cb').punishment.agg(['mean','size']).round(2))
print(df.assign(db=pd.cut(df.dev,[-21,-8,-4,-1,1,4,21])).groupby('db').punishment.agg(['mean','size']).round(2))
