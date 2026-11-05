import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-6')
from load import load
import numpy as np, pandas as pd
df=load()
# per game, group, round: pool; fill empty groups with 0
g=df.groupby(['game','round_number','group_id']).agg(cg=('common_good','first'),n=('player_id','size'),p=('punishment','sum'),c=('contribution','sum'),pn=('punishment',lambda x:(x>0).sum())).reset_index()
full=pd.MultiIndex.from_product([df.game.unique(),range(24),[0,1]],names=['game','round_number','group_id'])
g=g.set_index(['game','round_number','group_id']).reindex(full).fillna(0).reset_index()
chk=(1.6*g.c-g.p-g.cg).abs().max(); print('pool formula err',chk)
m=g.groupby(['game','group_id']).agg(pool=('cg','mean'),n=('n','mean'),p=('p','sum'),c=('c','sum'),pn=('pn','sum'),nr=('n','sum')).reset_index()
m['p_per']=m.p/m.nr.clip(1); m['c_per']=m.c/m.nr.clip(1); m['pfreq']=m.pn/m.nr.clip(1)
o=m[['game','group_id','p_per','pool','n','c_per']].copy(); o['group_id']=1-o.group_id
m=m.merge(o,on=['game','group_id'],suffixes=('','_o'))
m['dp']=m.p_per-m.p_per_o
print(m[['pool','n','c_per','p_per','pfreq','dp']].corr().round(2))
m['pb']=pd.qcut(m.p_per,5,duplicates='drop')
print(m.groupby('pb',observed=True)[['pool','n','c_per','pfreq']].mean().round(2))
m['dpb']=pd.cut(m.dp,[-10,-1,-0.3,0.3,1,10])
print(m.groupby('dpb',observed=True)[['pool','n','c_per']].agg(['mean','size']).round(2))
# first 4 rounds punishment vs later pool
e=g[g.round_number<4].groupby(['game','group_id']).p.sum().rename('p_early')
m=m.merge(e.reset_index(),on=['game','group_id'])
print(m[['p_early','pool','n']].corr().round(2))
m.to_pickle('/tmp/m.pkl')
