import pandas as pd, numpy as np
d=pd.read_csv('experiments/2group_8agent_50ep.csv').drop_duplicates(['session','episode','round_number','participant_code'])
d=d.sort_values(['session','episode','participant_code','round_number'])
k=['session','episode','participant_code']
d['g_next']=d.groupby(k).group_id.shift(-1)
d['sw']=(d.g_next!=d.group_id).astype(float)
dec=d[d.round_number.isin([3,7,11,15,19])].copy()
# window (last 4 rounds) features per player and group
d['win']=d.round_number//4
w=d.groupby(['session','episode','participant_code','win']).agg(p=('punishment','sum'),c=('contribution','mean'),pay=('payoff','mean'),g=('group_id','last'),sw=('sw','last')).reset_index()
gw=d.groupby(['session','episode','win','group_id']).agg(gp=('punishment','mean'),gc=('contribution','mean'),gpay=('payoff','mean'),n=('payoff','size')).reset_index()
gw['n']=gw.n/4
w=w.merge(gw,left_on=['session','episode','win','g'],right_on=['session','episode','win','group_id'])
o=gw.copy(); o['group_id']=1-o.group_id; o=o.rename(columns={'gp':'op','gc':'oc','gpay':'opay','n':'on'})
w=w.merge(o,left_on=['session','episode','win','g'],right_on=['session','episode','win','group_id'],how='left')
w=w[w.win<5]
print('switch rate',w.sw.mean(), 'by window', w.groupby('win').sw.mean().round(3).to_dict())
w['ownp_bin']=pd.cut(w.p,[-1,0,4,10,20,200])
print(w.groupby('ownp_bin',observed=True).sw.agg(['mean','size']).round(3))
w['dpay']=w.gpay-w.opay.fillna(0)
w['dpay_bin']=pd.qcut(w.dpay,6)
print(w.groupby('dpay_bin',observed=True).sw.agg(['mean','size']).round(3))
w['dgp']=w.gp-w.op.fillna(0)
w['dgp_bin']=pd.cut(w.dgp,[-30,-2,-0.5,0.5,2,30])
print(w.groupby('dgp_bin',observed=True).sw.agg(['mean','size']).round(3))
import statsmodels.formula.api as smf
w2=w.dropna(subset=['op'])
print(smf.logit('sw ~ p + gp + op + c + gc + oc + n',w2).fit(disp=0).summary2().tables[1].round(3))
