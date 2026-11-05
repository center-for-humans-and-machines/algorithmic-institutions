# manager style vs group outcome (per game and group)
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-7')
from load import load
import numpy as np, pandas as pd
df = load()
g = df.groupby(['game','round_number','group_id']).agg(c=('contribution','mean'), p=('punishment','mean'), n=('player_id','size'), pool=('common_good','first'),
   ppos=('punishment', lambda x:(x>0).mean())).reset_index()
# fill empty groups with pool 0
full = pd.MultiIndex.from_product([df.game.unique(), range(24), [0,1]], names=['game','round_number','group_id'])
g = g.set_index(['game','round_number','group_id']).reindex(full).reset_index()
g['pool']=g.pool.fillna(0); g['n']=g.n.fillna(0)
s = g.groupby(['game','group_id']).agg(pool=('pool','mean'), p=('p','mean'), c=('c','mean'), n=('n','mean'), ppos=('ppos','mean')).reset_index()
o = s.copy(); o['group_id']=1-o.group_id
s = s.merge(o, on=['game','group_id'], suffixes=('','_o'))
s['dp']=s.p-s.p_o
print(s[['pool','p','c','n','ppos','dp']].corr().round(2))
s['pq']=pd.qcut(s.p,5)
print(s.groupby('pq', observed=True)[['pool','c','n','p_o']].mean().round(2))
print(s.groupby(pd.qcut(s.dp,5), observed=True)[['pool','c','n']].mean().round(2))
import statsmodels.formula.api as smf
print(smf.ols('pool ~ p + p_o', s).fit().params.round(2))
print(smf.ols('n ~ dp', s).fit().params.round(2))
# per capita contribution vs group size
print(g.groupby('n').c.mean().round(2))
