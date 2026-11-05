import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-4')
from load import load
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
df, gs = load()
d = df[(df.player_no_input==0)&(df.next_valid==1)&(df.next_group==df.group_id)&(df.round_number<23)].copy()
d['dc'] = d.next_c - d.contribution
d['dev'] = d.contribution - d.loo_mc
print('dc by punishment bucket, split by own contribution')
d['cb'] = pd.cut(d.contribution,[-1,4,9,14,20])
d['pb'] = pd.cut(d.punishment,[-1,0,2,5,10,30])
print(d.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean').round(2))
print(d.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size'))
m = smf.ols('next_c ~ contribution + loo_mc + punishment + I(punishment**2) + round_number', d.dropna(subset=['loo_mc'])).fit(cov_type='cluster', cov_kwds={'groups': d.dropna(subset=['loo_mc']).game.astype('category').cat.codes})
print(m.summary().tables[1])
# low contributors only
lo = d[d.contribution<=8].dropna(subset=['loo_mc'])
m2 = smf.ols('next_c ~ contribution + loo_mc + punishment + round_number', lo).fit()
print('low contributors'); print(m2.params.round(3), m2.bse.round(3))
# persistence: effect of punishment at t on c at t+2,t+3 
df2 = df.copy(); g = df2.groupby(['game','player_id'])
for k in [2,3,4]:
    df2[f'c{k}'] = g['contribution'].shift(-k); df2[f'gr{k}'] = g['group_id'].shift(-k)
for k in [2,3,4]:
    e = df2[(df2[f'gr{k}']==df2.group_id)&(df2.player_no_input==0)].dropna(subset=['loo_mc',f'c{k}'])
    mm = smf.ols(f'c{k} ~ contribution + loo_mc + punishment + round_number', e).fit()
    print(k, 'punishment coef', round(mm.params.punishment,3), round(mm.bse.punishment,3))
