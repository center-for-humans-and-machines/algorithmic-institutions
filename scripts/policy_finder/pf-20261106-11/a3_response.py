import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from load import load
import pandas as pd, numpy as np
df=load()
g=df.groupby(['game','player_id'])
df['c_next']=g.contribution.shift(-1); df['grp_next']=g.group_id.shift(-1)
d=df[(df.grp_next==df.group_id)&(df.player_no_input==0)].dropna(subset=['c_next']).copy()
d['dc']=d.c_next-d.contribution
d['cb']=pd.cut(d.contribution,[-1,0,5,10,15,19,20])
d['pb']=pd.cut(d.punishment,[-1,0,2,5,10,31])
print('next-round change in contribution, by c and punishment')
print(d.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean',observed=True).round(2))
print(d.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size',observed=True))
# human managers: punishment as fn of contribution
print(df[df.player_no_input==0].groupby(pd.cut(df.contribution,[-1,0,5,10,14,19,20]),observed=True).punishment.agg(['mean','size']).round(2))
# group-level: avg punishment of manager in game-group vs mean contribution in rounds 12-23 and members
gg=df.groupby(['game','group_id','round_number']).agg(p=('punishment','mean'),c=('contribution','mean'),n=('player_id','size'),pool=('common_good','first')).reset_index()
early=gg[gg.round_number<8].groupby(['game','group_id']).p.mean().rename('p_early')
late=gg[gg.round_number>=8].groupby(['game','group_id']).agg(c_late=('c','mean'),n_late=('n','mean'),pool_late=('pool','mean'))
x=pd.concat([early,late],axis=1)
x['pb']=pd.qcut(x.p_early,4)
print(x.groupby('pb',observed=True).mean().round(2))
