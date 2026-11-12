import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d=d[d.run.str.contains('ah_vs_zero|zero_vs_ah')].copy()
d['mgr']=np.where(d.run.str.contains('ah_vs_zero'), np.where(d.agent_group==0,'ah','zero'), np.where(d.agent_group==0,'zero','ah'))
d=d.sort_values(['run','episode','participant_code','round_number'])
k=['run','episode','participant_code']
d['g_next']=d.groupby(k).agent_group.shift(-1)
for j in range(4):
    d[f'p_l{j}']=d.groupby(k).punishment.shift(j)
    d[f'c_l{j}']=d.groupby(k).contribution.shift(j)
gk=['run','episode','round_number','agent_group']
d['gp']=d.groupby(gk).punishment.transform('mean'); d['gc']=d.groupby(gk).contribution.transform('mean'); d['n']=d.groupby(gk).contribution.transform('size')
d['gp_oth']=d.gp-(d.punishment)/d.n
dec=d[d.round_number.isin([3,7,11,15,19])&(d.mgr=='ah')].dropna(subset=['p_l3']).copy()
dec['sw']=(dec.g_next!=dec.agent_group).astype(float)
m=smf.logit('sw ~ p_l0+p_l1+p_l2+p_l3+gp_oth+contribution+gc+n+C(round_number)',data=dec).fit(disp=0)
print(m.params.round(3)); print(m.bse.round(3))
