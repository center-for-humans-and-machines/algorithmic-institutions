"""Human data: contribution response to punishment, switching, manager behaviour."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
h=pd.read_csv('experiments/2group_8agent_50ep.csv')
h=h.drop_duplicates(['session','episode','round_number','participant_code'])
h['game']=h.session.astype(str)+'_'+h.episode.astype(str)
print('games',h.game.nunique(),'rows',len(h))
h=h.sort_values(['game','player_id','round_number'])
k=['game','player_id']
h['c_next']=h.groupby(k).contribution.shift(-1)
h['g_next']=h.groupby(k).group_id.shift(-1)
h['gmean']=h.groupby(['game','round_number','group_id']).contribution.transform('mean')
x=h[(h.g_next==h.group_id)&(h.player_no_input==0)&((h.round_number%4)!=3)].dropna(subset=['c_next'])
print(smf.ols('c_next ~ contribution + punishment + gmean + round_number',data=x).fit().params.round(3))
# switching
h['blk']=h.round_number//4
b=h.groupby(k+['blk']).agg(g=('group_id','first'),p=('punishment','mean'),c=('contribution','mean')).reset_index()
bn=b[k+['blk','g']].copy(); bn['blk']-=1; bn=bn.rename(columns={'g':'gn'})
b=b.merge(bn)
b['sw']=(b.g!=b.gn).astype(int)
print('switch rate',b.sw.mean())
b['pbin']=pd.cut(b.p,[-0.1,0,1,3,6,10,30])
print(b.groupby('pbin',observed=True).sw.agg(['mean','size']))
# group level: per game/group pool and size; manager punishment intensity per group-block
h['pool']=1.6*h.contribution-h.punishment
gg=h.groupby(['game','group_id']).agg(p=('punishment','mean')).reset_index()
pool=h.groupby(['game','round_number','group_id']).pool.sum().unstack(fill_value=0).groupby('game').mean()
gg=gg.pivot(index='game',columns='group_id',values='p')
z=pd.DataFrame({'dp':gg[0]-gg[1],'dpool':pool[0]-pool[1]})
print(z.corr()); print(smf.ols('dpool~dp',data=z).fit().params)
# who human managers punish
print(h.groupby(pd.cut(h.contribution,[-0.1,0,5,10,15,20]),observed=True).punishment.agg(['mean','size']))
print(h.groupby('round_number').agg(c=('contribution','mean'),p=('punishment','mean')).round(2).T.to_string())
