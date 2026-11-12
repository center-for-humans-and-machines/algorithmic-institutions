import pandas as pd, numpy as np
import statsmodels.formula.api as smf
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['pair']=d.run.str.split('by ').str[1]
d=d.sort_values(['pair','episode','participant_code','round_number'])
k=['pair','episode','participant_code']
d['mgr']=np.where(d.agent_group==0,d.pair.str.split('_vs_').str[0],d.pair.str.split('_vs_').str[1])
d['c_next']=d.groupby(k).contribution.shift(-1)
d['g_next']=d.groupby(k).agent_group.shift(-1)
gk=['pair','episode','round_number','agent_group']
d['gn']=d.groupby(gk).contribution.transform('size')
d['op']=(d.groupby(gk).punishment.transform('sum')-d.punishment)/(d.gn-1).clip(lower=1)
d['oc']=(d.groupby(gk).contribution.transform('sum')-d.contribution)/(d.gn-1).clip(lower=1)
a=d[(d.mgr=='ah')&(d.g_next==d.agent_group)&d.contribution_valid&(d.gn>1)].copy()
a['c0']=(a.contribution==0).astype(float)
a['lowp']=a.punishment*(a.contribution<=10); a['highp']=a.punishment*(a.contribution>10)
m=smf.ols('c_next ~ contribution + c0 + lowp + highp + op + oc + C(round_number)',a).fit()
print(m.params.filter(regex='^(contr|c0|lowp|highp|op|oc)').round(3), m.bse.filter(regex='^(lowp|highp|op|oc)').round(3))
# exact c=0 and c=2..4 comparisons
for cv in [0,2,5,8,10]:
    s=a[a.contribution==cv]
    print(cv, s.groupby(pd.cut(s.punishment,[-1,0,2,5,10,30]),observed=True).c_next.agg(['mean','size']).round(2).T.to_string())
# time trend of contributions by manager type
print(d[d.contribution_valid].groupby(['mgr','round_number']).contribution.mean().unstack().round(1).to_string())
