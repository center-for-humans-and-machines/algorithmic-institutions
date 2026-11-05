import numpy as np, pandas as pd
from load import load
df = load()
df['valid']=1-df.player_no_input
gr = df.groupby(['game','round_number','group_id']).agg(pool=('common_good','first'),n=('player_id','count'),pun=('punishment','sum'),c=('contribution','sum')).reset_index()
# full grid incl. empty groups
idx = pd.MultiIndex.from_product([gr.game.unique(),range(24),[0,1]],names=['game','round_number','group_id'])
gr = gr.set_index(['game','round_number','group_id']).reindex(idx,fill_value=0).reset_index()
ep = gr.groupby(['game','group_id']).agg(pool=('pool','mean'),n=('n','mean'),pun=('pun','sum'),c=('c','sum')).reset_index()
ep['pun_rate']=ep.pun/(ep.n*24)
print(ep.describe().round(2))
print(ep.groupby(pd.qcut(ep.pun_rate.rank(method='first'),5)).agg(pun_rate=('pun_rate','mean'),pool=('pool','mean'),n=('n','mean')).round(2))
print('corr', ep[['pool','n','pun_rate']].corr().round(2))
# within-game: higher punisher vs lower
w = ep.pivot(index='game',columns='group_id',values=['pool','pun_rate'])
hi = (w.pun_rate[0]>w.pun_rate[1])
print('pool of more-punishing group minus less', (np.where(hi,w.pool[0]-w.pool[1],w.pool[1]-w.pool[0])).mean().round(2))
# who do managers punish: by contribution
v=df[df.valid==1]
v['gm']=v.groupby(['game','round_number','group_id']).contribution.transform('mean')
print(v.groupby(pd.cut(v.contribution,[-1,0,4,9,14,19,20]),observed=True).punishment.agg(['mean',lambda x:(x>0).mean()]).round(2))
