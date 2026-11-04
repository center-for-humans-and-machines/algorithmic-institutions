"""Reference sims: does group punishment drive unpunished members out; absolute vs relative shortfall."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('managed by ah_vs_ah')].copy()
d['block']=d.round_number//4
d['gmean']=d.groupby(['episode','round_number','group_id']).contribution.transform('mean')
d['gpun']=d.groupby(['episode','round_number','group_id']).punishment.transform('mean')
d['n']=d.groupby(['episode','round_number','group_id']).participant_code.transform('size')
pl = d.groupby(['episode','participant_code','block']).agg(grp=('group_id','last'),p=('punishment','sum'),p_last=('punishment','last'),
     c=('contribution','mean'), gpun=('gpun','mean'), gc=('gmean','mean'), n=('n','last')).reset_index()
pl['gn']=pl.groupby(['episode','participant_code']).grp.shift(-1); pl=pl[pl.gn.notna()]; pl['left']=(pl.gn!=pl.grp).astype(int)
pl['others_p'] = (pl.gpun*pl.n*4 - pl.p)/(pl.n-1).clip(lower=1)/4
r=smf.logit('left ~ p_last + I(p-p_last) + others_p + c + gc + n', data=pl).fit(disp=0); print(r.params.round(4)); print(r.bse.round(4))
print(pl[pl.p==0].groupby(pd.cut(pl.others_p,[-1,0,0.5,1,2,4,30])).left.agg(['mean','count']).round(3))
# contribution response: absolute c vs deviation
d=d.sort_values(['episode','participant_code','round_number'])
g=d.groupby(['episode','participant_code'])
d['c2']=g.contribution.shift(-2); d['g2']=g.group_id.shift(-2)
w=d[(d.g2==d.group_id)&(d.contribution_valid==1)&(d.contribution<=12)].copy(); w['dc']=w.c2-w.contribution; w['dev']=w.contribution-w.gmean
r=smf.ols('dc ~ np.sqrt(punishment) + contribution + dev + np.sqrt(punishment):dev + np.sqrt(punishment):contribution + round_number', data=w).fit(); print(r.params.round(3))
