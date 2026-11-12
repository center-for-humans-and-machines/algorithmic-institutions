import pandas as pd, numpy as np
d=pd.read_csv('experiments/2group_8agent_50ep.csv').drop_duplicates(['session','episode','round_number','participant_code'])
d=d.sort_values(['session','episode','participant_code','round_number'])
print(len(d), d.groupby(['session','episode']).ngroups)
d['pool']=1.6*d.contribution-d.punishment
k=['session','episode','participant_code']
d['c_next']=d.groupby(k).contribution.shift(-1)
d['g_next']=d.groupby(k).group_id.shift(-1)
d['dc']=d.c_next-d.contribution
# group-level
G=d.groupby(['session','episode','round_number','group_id']).agg(n=('pool','size'),pool=('pool','sum'),c=('contribution','mean'),p=('punishment','mean'),pay=('payoff','mean')).reset_index()
print(G.groupby('round_number')[['n','c','p']].mean().round(2).T.to_string())
# punishment policy of humans: punishment vs contribution
d['cbin']=pd.cut(d.contribution,[-1,0,5,10,15,19,20])
print(d.groupby('cbin').punishment.agg(['mean','size',lambda x:(x>0).mean()]).round(2))
# response: next contribution change by punishment received, controlling for contribution bin
d['pbin']=pd.cut(d.punishment,[-1,0,2,5,10,30])
s=d[d.g_next==d.group_id]
print(s.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean').round(2))
print(s.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='size'))
