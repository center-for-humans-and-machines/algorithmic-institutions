"""Does punishment raise next-round contributions in the sims (ah-managed players)?"""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d=d.sort_values(['run','episode','participant_code','round_number'])
k=['run','episode','participant_code']
d['c_next']=d.groupby(k).contribution.shift(-1)
d['g_next']=d.groupby(k).agent_group.shift(-1)
d['valid_next']=d.groupby(k).contribution_valid.shift(-1)
gm=d.groupby(['run','episode','round_number','agent_group']).contribution.transform('mean')
d['gmean']=gm
x=d[(d.g_next==d.agent_group)&d.contribution_valid&(d.valid_next==True)&((d.round_number%4)!=3)].copy()
x['dc']=x.c_next-x.contribution
x['pbin']=pd.cut(x.punishment,[-0.1,0,2,5,10,30])
x['cbin']=pd.cut(x.contribution,[-0.1,0,5,10,15,20])
print(x.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean',observed=True).round(2))
print(x.pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='size',observed=True))
m=smf.ols('c_next ~ contribution + punishment + gmean + round_number',data=x).fit()
print(m.params.round(3))
# effect on longer horizon: c in round+2,+3
for h in [2,3]:
    x[f'c{h}']=d.groupby(k).contribution.shift(-h).loc[x.index]
    print(h, smf.ols(f'c{h} ~ contribution + punishment + gmean + round_number',data=x.dropna(subset=[f'c{h}'])).fit().params.round(3).to_dict())
