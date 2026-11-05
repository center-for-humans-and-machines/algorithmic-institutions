import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d = d[d.run.str.endswith('humanlike')].copy()
d['valid']=d.contribution_valid.astype(int); d.loc[d.valid==0,'contribution']=0
d=d.sort_values(['episode','participant_code','round_number']); g=d.groupby(['episode','participant_code'])
# future punishments received in next 5 rounds (confound)
d['pfut'] = sum(g.punishment.shift(-k).fillna(0) for k in range(1,6))
for k in [1,5]:
    d['cn']=g.contribution.shift(-k); d['vn']=g.valid.shift(-k)
    x=d[(d.valid==1)&(d.vn==1)].copy(); x['dc']=x.cn-x.contribution
    x['cbin']=pd.cut(x.contribution,[-1,4,9,14,19,20]); x['pc']=x.punishment.clip(upper=12)
    print(f'\n+{k} dc by exact p (cols) and c bin'); print(x.pivot_table(index='cbin',columns='pc',values='dc',observed=False).round(2).iloc[:, [0,1,2,3,4,5,6,8,10,12]])
x['early']=x.round_number<8
print('\nfuture punishment over next 5 rounds by current p, c<10'); print(x[x.contribution<10].groupby('pc').pfut.mean().round(1))
print('\nround split, c 5-14, +5 dc by p>0'); x['pp']=x.punishment>0
print(x[(x.contribution>=5)&(x.contribution<15)].pivot_table(index=pd.cut(x.round_number,[-1,5,11,18]),columns='pp',values='dc',observed=False).round(2))
