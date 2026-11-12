import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261112-3')
from human_load import load
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
d=load()
d['c_next']=d.groupby(['game','player_id']).contribution.shift(-1)
d['g_next']=d.groupby(['game','player_id']).group_id.shift(-1)
gs=d.groupby(['game','round_number','group_id']).contribution.transform('mean')
d['gmean']=gs
x=d[(d.player_no_input==0)&d.c_next.notna()&(d.g_next==d.group_id)&~d.round_number.isin([3,7,11,15,19])].copy()
x['dc']=x.c_next-x.contribution
x['cbin']=pd.cut(x.contribution,[-1,4,9,14,19,20])
x['pbin']=pd.cut(x.punishment,[-1,0,2,5,10,30])
print('mean next-round change dc by own contribution bin x punishment bin')
print(x.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean').round(2))
print(x.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='size'))
import statsmodels.formula.api as smf
x['rel']=x.contribution-x.gmean
m=smf.ols('dc ~ punishment*contribution + rel + round_number',data=x).fit()
print(m.params.round(3), m.bse.round(3))
# what human managers did
print('human manager punishment by contribution bin')
h=d[(d.player_no_input==0)&(d.manager_no_input==0)].copy()
h['cb']=pd.cut(h.contribution,[-1,0,4,9,14,19,20])
print(h.groupby('cb').punishment.agg(['mean','size']).round(2))
h['rb']=pd.cut(h.contribution-h.gmean,[-21,-8,-4,-1,1,4,21])
print(h.groupby('rb').punishment.agg(['mean','size']).round(2))
print(h.groupby('round_number').punishment.mean().round(2).tolist())
