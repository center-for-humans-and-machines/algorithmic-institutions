import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
print(d.columns.tolist(), d.shape)
d['pl'] = d.participant_code.str.split('_').str[0]
d = d.sort_values(['run','episode','pl','round_number'])
d.loc[~d.contribution_valid,'contribution']=0
g = d.groupby(['run','episode','pl'])
d['c_next']=g.contribution.shift(-1); d['v_next']=g.contribution_valid.shift(-1)
s = d.groupby(['run','episode','round_number']).contribution.transform('sum')
nv = d.groupby(['run','episode','round_number']).contribution_valid.transform('sum')
d['others']=(s-d.contribution)/(nv-d.contribution_valid.astype(int)).clip(lower=1)
print('timeout rate', d.groupby('run').contribution_valid.apply(lambda v:1-v.mean()))
x = d[d.contribution_valid & (d.v_next==True)].copy()
x['dc']=x.c_next-x.contribution
x['cb']=pd.cut(x.contribution,[-1,0,4,9,14,19,20]); x['pb']=pd.cut(x.punishment,[-1,0,2,5,10,30])
for r in x.run.unique():
    z=x[x.run==r]; print(r)
    print(z.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean',observed=True).round(2))
    print(z.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size',observed=True))
    X=np.c_[np.ones(len(z)), z.punishment, z.contribution, z.others-z.contribution]
    print('dc~1,p,c,others-c', np.linalg.lstsq(X,z.dc.values,rcond=None)[0].round(3))
h = x[x.run.str.contains('humanlike')]
print('humanlike punish by c bin', h.groupby('cb',observed=True).punishment.agg(['mean',lambda v:(v>0).mean()]).round(2))
