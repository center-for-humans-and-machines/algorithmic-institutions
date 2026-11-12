import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d=d[d.run.str.contains('ah_vs_zero|zero_vs_ah')].copy()
d['mgr']=np.where(d.run.str.contains('ah_vs_zero'), np.where(d.agent_group==0,'ah','zero'), np.where(d.agent_group==0,'zero','ah'))
d=d.sort_values(['run','episode','participant_code','round_number'])
k=['run','episode','participant_code']
gk=['run','episode','round_number','agent_group']
d['gc']=d.groupby(gk).contribution.transform('mean'); d['n']=d.groupby(gk).contribution.transform('size')
d['gp_oth']=(d.groupby(gk).punishment.transform('sum')-d.punishment)/(d.n-1).clip(lower=1)
d['c_prev']=d.groupby(k).contribution.shift(1)
for h in [1,2,3]:
    d[f'c_f{h}']=d.groupby(k).contribution.shift(-h)
    d[f'g_f{h}']=d.groupby(k).agent_group.shift(-h)
a=d[(d.mgr=='ah')&d.contribution_valid.astype(bool)].copy()
a['rel']=a.contribution-a.gc
a['p_x_c']=a.punishment*a.contribution
for h in [1,2,3]:
    x=a[(a[f'g_f{h}']==a.agent_group)].dropna(subset=[f'c_f{h}','c_prev'])
    m=smf.ols(f'c_f{h} ~ punishment + p_x_c + contribution + c_prev + gc + gp_oth + round_number',data=x).fit()
    print(h, m.params[['punishment','p_x_c','gp_oth','gc']].round(3).to_dict())
