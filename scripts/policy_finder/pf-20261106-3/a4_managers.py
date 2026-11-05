import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-3')
from load import load
import numpy as np, pandas as pd
df = load()
gr = df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'),c=('contribution','sum'),p=('punishment','sum'),cg=('common_good','first'),pun_low=('punishment',lambda s: 0)).reset_index()
# fill empty groups
idx = pd.MultiIndex.from_product([df.game.unique(),range(24),[0,1]],names=['game','round_number','group_id'])
gr = gr.set_index(['game','round_number','group_id']).reindex(idx).fillna(0).reset_index()
ep = gr.groupby(['game','group_id']).agg(pool=('cg','mean'),n=('n','mean'),c=('c','sum'),p=('p','sum'),nsum=('n','sum')).reset_index()
ep['pmean']=ep.p/ep.nsum.clip(1); ep['cmean']=ep.c/ep.nsum.clip(1)
o = ep[['game','group_id','pmean','pool','n']].copy(); o['group_id']=1-o.group_id
ep = ep.merge(o,on=['game','group_id'],suffixes=('','_o'))
print(ep[['pool','n','pmean','cmean','pmean_o']].corr().round(2))
ep['dpun']=ep.pmean-ep.pmean_o
print(ep.groupby(pd.qcut(ep.dpun,5)).agg(dpun=('dpun','mean'),pool=('pool','mean'),n=('n','mean'),cmean=('cmean','mean')).round(2))
print(ep.groupby(pd.qcut(ep.pmean,5)).agg(pm=('pmean','mean'),pool=('pool','mean'),n=('n','mean'),cmean=('cmean','mean')).round(2))
# decomposition: var of pool explained by n
import statsmodels.formula.api as smf
print(smf.ols('pool ~ n + cmean + pmean',ep).fit().params.round(2))
# per-capita pool vs punishment in rounds 0-3 only (before switching)
f = df[df.round_number<4]
fg = f.groupby(['game','group_id']).agg(p=('punishment','mean'),c=('contribution','mean')).reset_index()
l = df[df.round_number>=4].groupby(['game','group_id']).agg(n=('player_id','size')).reset_index(); l['n']/=20
fg = fg.merge(l,on=['game','group_id'],how='left').fillna(0)
o=fg[['game','group_id','p','c']].copy(); o['group_id']=1-o.group_id
fg=fg.merge(o,on=['game','group_id'],suffixes=('','_o'))
print(smf.ols('n ~ p + c + p_o + c_o',fg).fit().summary().tables[1])
