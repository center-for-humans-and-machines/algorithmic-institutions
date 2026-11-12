import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d=d[d.run.str.contains('ah_vs_zero|zero_vs_ah|ah_vs_ah')].copy()
d['ep']=d.run+d.episode.astype(str)
d=d.sort_values(['ep','participant_code','round_number'])
g=d.groupby(['ep','participant_code'])
d['nc']=g.contribution.shift(-1); d['ng']=g.agent_group.shift(-1)
d['run_s']=d.run.str.split('by ').str[-1]
mgr=np.where(d.agent_group==0,d.run_s.str.split('_vs_').str[0],d.run_s.str.split('_vs_').str[1])
d['mgr']=mgr
x=d[(d.ng==d.agent_group)&(d.contribution_valid)&(d.mgr=='ah')].dropna(subset=['nc']).copy()
x['dc']=x.nc-x.contribution
x['pb']=pd.cut(x.punishment,[-.1,0,2,5,10,31]); x['cb']=pd.cut(x.contribution,[-.1,4,9,14,19,20])
print(x.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean',observed=False).round(2))
print(x.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size',observed=False))
# switching in sims
d['blk']=d.round_number//4
d['sw']=(d.ng!=d.agent_group).astype(float)
dec=d[d.round_number.isin([3,7,11,15,19])].copy()
bl=d.groupby(['ep','blk','participant_code']).agg(op=('punishment','mean'),grp=('agent_group','last'),oc=('contribution','mean')).reset_index()
gp=d.groupby(['ep','blk','agent_group']).agg(gp=('punishment','mean'),gc=('contribution','mean'),gn=('participant_code','nunique')).reset_index()
bl=bl.merge(gp,left_on=['ep','blk','grp'],right_on=['ep','blk','agent_group'])
gp2=gp.rename(columns={'agent_group':'oth','gp':'xp','gc':'xc','gn':'xn'})
bl['oth']=1-bl.grp
bl=bl.merge(gp2,on=['ep','blk','oth'],how='left').fillna({'xp':0,'xc':0,'xn':0})
dec=dec.merge(bl[['ep','blk','participant_code','op','oc','gp','gc','gn','xp','xc','xn']],on=['ep','blk','participant_code'])
dec['dp']=dec.gp-dec.xp; dec['dcpool']=1.6*(dec.gc*dec.gn-dec.xc*dec.xn)
print('switch rate',dec.sw.mean(), 'by mgr', dec.groupby('mgr').sw.mean().round(3).to_dict())
for col,bins in [('op',[-.1,0,1,3,6,31]),('dp',[-31,-2,-.5,.5,2,31]),('gn',[0,2,3,4,5,6,8]),('oc',[-.1,4,9,14,21])]:
    print(dec.groupby(pd.cut(dec[col],bins),observed=False).sw.agg(['mean','size']).round(3))
import statsmodels.formula.api as smf
m=smf.logit('sw ~ op + dp + gn + oc + gc + xc',data=dec).fit(disp=0); print(m.summary().tables[1])
