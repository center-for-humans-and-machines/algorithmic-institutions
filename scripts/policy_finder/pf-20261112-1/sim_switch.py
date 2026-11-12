import pandas as pd, numpy as np
import statsmodels.formula.api as smf
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['pair']=d.run.str.split('by ').str[1]
d=d.sort_values(['pair','episode','participant_code','round_number'])
k=['pair','episode','participant_code']
d['g_next']=d.groupby(k).agent_group.shift(-1)
d['sw']=(d.g_next!=d.agent_group).astype(float)
d['win']=d.round_number//4
d['mgr']=np.where(d.agent_group==0,d.pair.str.split('_vs_').str[0],d.pair.str.split('_vs_').str[1])
d['p_last']=np.where(d.round_number%4==3,d.punishment,0)
w=d.groupby(k+['win']).agg(p=('punishment','sum'),plast=('p_last','sum'),c=('contribution','mean'),g=('agent_group','last'),sw=('sw','last'),mgr=('mgr','last')).reset_index()
gw=d.groupby(['pair','episode','win','agent_group']).agg(gp=('punishment','mean'),gc=('contribution','mean'),n=('payoff','size')).reset_index()
gw['n']=gw.n/4
w=w.merge(gw,left_on=['pair','episode','win','g'],right_on=['pair','episode','win','agent_group'])
o=gw.rename(columns={'gp':'op','gc':'oc','n':'on'}); o['agent_group']=1-o.agent_group
w=w.merge(o,left_on=['pair','episode','win','g'],right_on=['pair','episode','win','agent_group'],how='left')
w=w[w.win<5]
print(w.groupby(['mgr','win']).sw.mean().unstack().round(3))
w['pb']=pd.cut(w.p,[-1,0,4,10,20,200])
print(w[w.mgr=='ah'].groupby(['pair','pb'],observed=True).sw.agg(['mean','size']).round(3))
w2=w.dropna(subset=['op'])
m=smf.logit('sw ~ p + plast + gp + op + c + gc + oc + n + C(win)',w2).fit(disp=0)
print(m.summary2().tables[1].round(3))
