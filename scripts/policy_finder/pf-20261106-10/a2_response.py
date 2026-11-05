# How does next-round contribution respond to this round's punishment, by contribution level?
import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-10')
import pandas as pd, numpy as np
from load import load
df=load()
d=df[df.next_c.notna() & (df.player_no_input==0)].copy()
d['same']=d.next_group==d.group_id
d=d[d.same]
d['dc']=d.next_c-d.contribution
d['cb']=pd.cut(d.contribution,[-1,4,9,14,19,20])
d['pb']=pd.cut(d.punishment,[-1,0,2,5,10,30])
print(d.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean').round(2))
print(d.pivot_table(index='cb',columns='pb',values='dc',aggfunc='count'))
# relative contribution: own minus group mean
d['gm']=d.groupby(['game','round_number','group_id']).contribution.transform('mean')
d['rel']=pd.cut(d.contribution-d.gm,[-21,-5,-1,1,5,21])
print(d.pivot_table(index='rel',columns='pb',values='dc',aggfunc='mean').round(2))
# does punishment of OTHERS in group affect your contribution (spillover)?
d['op']=(d.groupby(['game','round_number','group_id']).punishment.transform('sum')-d.punishment)/(d['size']-1).clip(lower=1)
d['opb']=pd.cut(d.op,[-1,0,1,3,30])
print(d[d.punishment==0].groupby(['cb','opb']).dc.mean().unstack().round(2))
