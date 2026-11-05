import numpy as np, pandas as pd
from load import load
df = load().sort_values(['game','player_id','round_number'])
g = df.groupby(['game','player_id'])
df['c_next'] = g.contribution.shift(-1)
df['grp_next'] = g.group_id.shift(-1)
df['valid'] = 1-df.player_no_input
d = df[(df.valid==1)&df.c_next.notna()&(df.grp_next==df.group_id)].copy()
d['dc'] = d.c_next-d.contribution
d['pbin'] = pd.cut(d.punishment,[-1,0,2,5,10,30])
d['cbin'] = pd.cut(d.contribution,[-1,4,9,14,19,20])
print('mean contribution by round'); print(df[df.valid==1].groupby('round_number').contribution.mean().round(1).to_string())
print('\nnext-round change dc by contribution bin x punishment bin')
print(d.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean').round(2))
print(d.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='count'))
# relative to group mean
d['gmean'] = d.groupby(['game','round_number','group_id']).contribution.transform('mean')
d['rel'] = pd.cut(d.contribution-d.gmean,[-21,-5,-1,1,5,21])
print('\ndc by rel to group mean x punished')
print(d.pivot_table(index='rel',columns=d.punishment>0,values='dc',aggfunc=['mean','count']).round(2))
