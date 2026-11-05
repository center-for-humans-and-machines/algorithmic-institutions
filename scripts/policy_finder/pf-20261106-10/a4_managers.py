# Per (game, group): mean punishment vs mean pool, size, contribution. Also punish style.
import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-10')
import pandas as pd, numpy as np
from load import load
df=load()
full=pd.MultiIndex.from_product([df.game.unique(),range(24),[0,1]],names=['game','round_number','group_id'])
gr=df.groupby(['game','round_number','group_id']).agg(pool=('common_good','first'),size=('player_id','count'),pun=('punishment','sum'),c=('contribution','mean')).reindex(full).fillna({'pool':0,'size':0,'pun':0})
g=gr.groupby(['game','group_id']).agg(pool=('pool','mean'),size=('size','mean'),pun=('pun','mean'),c=('c','mean'))
g['pun_per_member']=g.pun/g['size']
g['pb']=pd.qcut(g.pun,4,duplicates='drop')
print(g.groupby('pb',observed=True)[['pool','size','c','pun']].mean().round(2))
print(g[['pool','size','pun','c']].corr().round(2))
# within-game: group with less punishment vs more
w=g.reset_index().pivot(index='game',columns='group_id')
lessp=np.where(w.pun[0]<w.pun[1],0,1)
pl=np.where(lessp==0,w.pool[0],w.pool[1]); pm=np.where(lessp==0,w.pool[1],w.pool[0])
print('less-punishing group pool',pl.mean(),'more',pm.mean())
# punishment rule shape: P(punish) and mean by contribution
df['cb']=pd.cut(df.contribution,[-1,0,4,9,14,19,20])
print(df.groupby('cb',observed=True).punishment.agg(['mean',lambda x:(x>0).mean()]).round(2))
df['gm']=df.groupby(['game','round_number','group_id']).contribution.transform('mean')
df['rel']=pd.cut(df.contribution-df.gm,[-21,-8,-4,-1,1,4,21])
print(df.groupby('rel',observed=True).punishment.agg(['mean',lambda x:(x>0).mean()]).round(2))
print('--- within game, more- vs less-punishing group')
for k in ['size','c','pun']:
    a=np.where(lessp==0,w[k][0],w[k][1]); b=np.where(lessp==0,w[k][1],w[k][0])
    print(k,'less',np.nanmean(a).round(2),'more',np.nanmean(b).round(2))
print('games where more-punishing has larger pool', (pm>pl).mean())
# manager timeouts / zero-punishment managers
z=g.reset_index(); print('groups with pun<0.5/round:',(z.pun<0.5).sum(), z[z.pun<0.5][['pool','size','c']].mean().round(2).to_dict())
print('--- per-member punishment, within game')
pm_=df.groupby(['game','group_id']).punishment.mean().unstack()
lp=np.where(pm_[0]<pm_[1],0,1)
for k in ['pool','size','c']:
    a=np.where(lp==0,w[k][0],w[k][1]); b=np.where(lp==0,w[k][1],w[k][0])
    print(k,'less',np.nanmean(a).round(2),'more',np.nanmean(b).round(2), 'frac more>less', np.mean(b>a).round(2))
print('pun/member less',np.where(lp==0,pm_[0],pm_[1]).mean().round(2),'more',np.where(lp==0,pm_[1],pm_[0]).mean().round(2))
