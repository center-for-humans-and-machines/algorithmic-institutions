from load import load
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
df=load().sort_values(['game','player_id','round_number'])
grp=df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'),cm=('contribution','mean'),pm=('punishment','mean')).reset_index()
df=df.merge(grp,on=['game','round_number','group_id'])
oth=grp.rename(columns={'n':'on','cm':'ocm','pm':'opm'}); oth['group_id']=1-oth.group_id
df=df.merge(oth,on=['game','round_number','group_id'],how='left').fillna({'on':0,'ocm':0,'opm':0})
df=df.sort_values(['game','player_id','round_number'])
g=df.groupby(['game','player_id'])
for k in [1,2,3,4]:
    df[f'c{k}']=g.contribution.shift(-k); df[f'g{k}']=g.group_id.shift(-k)
df['gap']=(df.cm-df.contribution)
d=df[(df.player_no_input==0)&(df.gap>0)]
for k in [1,2,3,4]:
    s=d[(d[f'g{k}']==d.group_id)]
    r=smf.ols(f'c{k} ~ contribution + cm + punishment + round_number',s).fit()
    print(k,'below-mean players: punishment coef',r.params.punishment.round(3),'+-',r.bse.punishment.round(3),'n',len(s))
d=df[(df.player_no_input==0)&(df.gap<=0)]
s=d[(d.g1==d.group_id)]
r=smf.ols('c1 ~ contribution + cm + punishment + round_number',s).fit(); print('at/above mean: p coef',r.params.punishment.round(3),r.bse.punishment.round(3))
# switching interaction with other group's punishment
df['blk']=df.round_number//4
b=df.groupby(['game','player_id','blk']).agg(bp=('punishment','mean'),bpm=('pm','mean'),bopm=('opm','mean'),bcm=('cm','mean'),bocm=('ocm','mean'),n=('n','last'),c=('contribution','mean'),grp=('group_id','last')).reset_index()
nx=df[df.round_number%4==0][['game','player_id','blk','group_id']].copy(); nx['blk']-=1
b=b.merge(nx.rename(columns={'group_id':'ng'}),on=['game','player_id','blk'])
b['sw']=(b.ng!=b.grp).astype(int)
r=smf.logit('sw ~ bp*bopm + bpm + bcm + bocm + n + c',b).fit(disp=0); print(r.summary2().tables[1].round(3))
b['ob']=pd.cut(b.bopm,[-1,0.5,2,30]); b['pb']=pd.cut(b.bp,[-1,0,3,30])
print(b.pivot_table(index='pb',columns='ob',values='sw',aggfunc=['mean','size']).round(2))
