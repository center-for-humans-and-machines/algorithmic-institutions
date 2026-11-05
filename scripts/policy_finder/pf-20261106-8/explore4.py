from load import load
import pandas as pd, numpy as np
df=load()
grp=df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'),cm=('contribution','mean'),cmax=('contribution','max')).reset_index()
df=df.merge(grp,on=['game','round_number','group_id'])
ep=df.groupby(['game','group_id']).agg(pm=('punishment','mean'),pool=('common_good','sum')).reset_index()
ep['q']=pd.qcut(ep.pm.rank(method='first'),4,labels=False)
df=df.merge(ep[['game','group_id','q']],on=['game','group_id'])
df=df[df.player_no_input==0]
df['gap']=df.cm-df.contribution
df['gb']=pd.cut(df.gap,[-21,-3,0,3,6,10,21])
df['cb']=pd.cut(df.contribution,[-1,3,7,11,15,19,20])
for q in range(4):
    s=df[df.q==q]
    print('Q',q); print(s.groupby('cb').punishment.agg(['mean',lambda x:(x>0).mean(),'size']).round(2).T)
    print(s.groupby('gb').punishment.agg(['mean',lambda x:(x>0).mean()]).round(2).T)
    print('by phase',s.groupby(s.round_number//4).punishment.mean().round(2).to_dict())
