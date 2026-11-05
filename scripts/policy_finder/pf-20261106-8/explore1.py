from load import load
import pandas as pd, numpy as np
df=load().sort_values(['game','player_id','round_number'])
g=df.groupby(['game','player_id'])
df['next_c']=g.contribution.shift(-1)
df['next_group']=g.group_id.shift(-1)
df['dc']=df.next_c-df.contribution
# group-level
grp=df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'),cmean=('contribution','mean'),pmean=('punishment','mean'),cg=('common_good','first')).reset_index()
print('by round: mean c, mean p, mean group size');print(df.groupby('round_number')[['contribution','punishment']].mean().round(2).T)
print(grp.groupby('round_number').n.apply(lambda s:(s.max(),s.min(),s.std())).head(0))
# group size distribution at end
print('group sizes round 23', grp[grp.round_number==23].n.value_counts().sort_index().to_dict())
# group mean contribution
df=df.merge(grp[['game','round_number','group_id','n','cmean','pmean']],on=['game','round_number','group_id'])
df['rel']=df.contribution-df.cmean
# response to punishment, non-switch rows, same group next round
m=(df.next_group==df.group_id)&df.next_c.notna()&(df.player_no_input==0)
d=df[m].copy()
d['pbin']=pd.cut(d.punishment,[-1,0,2,5,10,30])
d['cbin']=pd.cut(d.contribution,[-1,4,9,14,19,20])
print('mean dc by contribution bin x punishment bin');print(d.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean').round(2))
print(d.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='size'))
import statsmodels.formula.api as smf
r=smf.ols('next_c ~ contribution + cmean + punishment + I(punishment**2)+round_number',d).fit(); print(r.params.round(3)); print(r.bse.round(3))
