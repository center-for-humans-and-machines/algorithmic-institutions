import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-1')
from load import load
import pandas as pd, numpy as np
df = load()
d = df[(df.next_c.notna()) & (df.player_no_input==0) & (df.switch==0)].copy()
d['dc'] = d.next_c - d.contribution
d['cbin']=pd.cut(d.contribution,[-1,0,5,10,15,19,20])
d['pbin']=pd.cut(d.punishment,[-1,0,2,5,10,30])
print(d.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean',observed=False).round(2))
print(d.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='size',observed=False))
# relative to group mean
d['gmean']=d.groupby(['game','round_number','group_id']).contribution.transform('mean')
d['rel']=d.contribution-d.gmean
import statsmodels.formula.api as smf
m=smf.ols('next_c ~ contribution + punishment + I(punishment*(contribution<10)) + gmean + round_number',d).fit()
print(m.params.round(3), m.bse.round(3))
# lasting effect: c at t+2..t+4 after punishment
g = df.groupby(['game','player_id'])
for k in [2,3,4]:
    df[f'c{k}']=g.contribution.shift(-k)
dd=df[(df.player_no_input==0)]
m2=smf.ols('c4 ~ contribution + punishment + round_number',dd).fit(); print('4 ahead', m2.params.round(3).to_dict())
# group-level: within-game, does group mean punishment relate to next block contribution
print('fraction 0/20 contributions', (df.contribution==0).mean(), (df.contribution==20).mean())
# how human managers punish: punishment vs contribution and vs group mean
df['gmean']=df.groupby(['game','round_number','group_id']).contribution.transform('mean')
df['below']=(df.gmean-df.contribution).clip(lower=0)
print(df.groupby(pd.cut(df.contribution,[-1,0,5,10,15,19,20]),observed=False).punishment.agg(['mean','size']).round(2))
m3=smf.ols('punishment ~ contribution + below + round_number',df[df.manager_no_input==0]).fit(); print(m3.params.round(3).to_dict())
