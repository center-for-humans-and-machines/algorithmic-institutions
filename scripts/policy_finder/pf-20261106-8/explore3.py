from load import load
import pandas as pd, numpy as np
df=load()
grp=df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'),c=('contribution','sum'),p=('punishment','sum'),cg=('common_good','first'),npun=('punishment',lambda s:(s>0).sum())).reset_index()
# fill missing groups as 0 pool
full=pd.MultiIndex.from_product([grp.game.unique(),range(24),[0,1]],names=['game','round_number','group_id'])
grp=grp.set_index(['game','round_number','group_id']).reindex(full).fillna(0).reset_index()
grp['pool']=1.6*grp.c-grp.p
print('check pool formula', np.corrcoef(grp.pool,grp.cg)[0,1])
ep=grp.groupby(['game','group_id']).agg(pool=('pool','mean'),n=('n','mean'),p=('p','mean'),c=('c','mean'),fracpun=('npun','sum')).reset_index()
ep['p_per_member']=ep.p/ep.n.clip(lower=0.1)
ep['frac_pun']=ep.fracpun/(grp.groupby(['game','group_id']).n.sum().values)
print(ep.describe().round(2))
print(ep[['pool','n','p_per_member','frac_pun']].corr().round(2))
ep['pq']=pd.qcut(ep.p_per_member.rank(method='first'),4,labels=False)
print(ep.groupby('pq')[['p_per_member','frac_pun','pool','n','c']].mean().round(2))
# pairwise: within a game, group with less punishment gets more members?
w=ep.pivot(index='game',columns='group_id',values=['pool','p_per_member','n'])
lessp=(w['p_per_member'][0]<w['p_per_member'][1])
pool_less=np.where(lessp,w['pool'][0],w['pool'][1]); pool_more=np.where(lessp,w['pool'][1],w['pool'][0])
print('within game: pool of less-punishing group',pool_less.mean().round(1),'more-punishing',pool_more.mean().round(1),'share less wins',(pool_less>pool_more).mean())
