# Fine shape of the AH response to punishment, 1 and 3 rounds ahead, by contribution level
import pandas as pd, numpy as np
pd.set_option('display.width',200)
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d=d[d.run.str.contains('humanlike')].copy()
d['pl']=d.participant_code.str.split('_').str[0]
d.loc[~d.contribution_valid,'contribution']=0
d=d.sort_values(['episode','pl','round_number']); g=d.groupby(['episode','pl'])
for k in (1,3): d[f'c{k}']=g.contribution.shift(-k); d[f'v{k}']=g.contribution_valid.shift(-k)
x=d[d.contribution_valid&(d.v3==True)&(d.round_number<=18)].copy()
x['pb']=pd.cut(x.punishment,[-1,0,3,6,9,12,15,20,30])
x['cb']=pd.cut(x.contribution,[-1,3,8,12,16,20])
for k in (1,3):
    x['dc']=x[f'c{k}']-x.contribution
    t=x.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean',observed=True)
    print(f'k={k} gain over unpunished, per bin'); print(t.sub(t.iloc[:,0],axis=0).round(2))
    print('per unit'); mp=x.groupby('pb',observed=True).punishment.mean()
    print((t.sub(t.iloc[:,0],axis=0)/mp).round(3))
print(x.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size',observed=True))
