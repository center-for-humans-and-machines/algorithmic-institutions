import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d = d[d.run.str.endswith('humanlike')].copy()
d['valid']=d.contribution_valid.astype(int); d.loc[d.valid==0,'contribution']=0
d=d.sort_values(['episode','participant_code','round_number']); g=d.groupby(['episode','participant_code'])
s = d[d.valid==1].groupby(['episode','round_number']).contribution.agg(['sum','count']); d=d.join(s,on=['episode','round_number'])
d['oth']=((d['sum']-d.contribution*d.valid)/(d['count']-d.valid).clip(lower=1)).round()
d['othb']=pd.cut(d.oth,[-1,7,13,20])
d['cn']=g.contribution.shift(-1); d['vn']=g.valid.shift(-1)
x=d[(d.valid==1)&(d.vn==1)&d.contribution.isin([0,5,10,15,18,20])].copy(); x['dc']=x.cn-x.contribution
x['pc']=x.punishment.clip(upper=8)
t=x.pivot_table(index=['contribution','othb'],columns='pc',values='dc',observed=False).round(2)
n=x.pivot_table(index=['contribution','othb'],columns='pc',values='dc',aggfunc='count',observed=False)
print(t.where(n>=40))
