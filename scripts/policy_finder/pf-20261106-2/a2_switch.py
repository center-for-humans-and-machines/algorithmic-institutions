import numpy as np, pandas as pd
from load import load
df = load().sort_values(['game','player_id','round_number'])
g = df.groupby(['game','player_id'])
df['grp_next'] = g.group_id.shift(-1)
dec = df[df.round_number.isin([3,7,11,15,19])].copy()
dec['switch'] = (dec.grp_next!=dec.group_id).astype(int)
# own block (4 rounds) punishment sum, group pool mean, other group's pool mean
df['block'] = df.round_number//4
gp = df.groupby(['game','block','group_id']).agg(pool=('common_good','mean'), pun=('punishment','mean'), n=('player_id','count')).reset_index()
gp['n']=gp.n/4
pb = df.groupby(['game','block','player_id']).agg(mypun=('punishment','sum'), myc=('contribution','mean'), lastp=('punishment','last')).reset_index()
dec['block']=dec.round_number//4
dec = dec.merge(pb,on=['game','block','player_id'])
dec = dec.merge(gp,on=['game','block','group_id'])
o = gp.copy(); o['group_id']=1-o.group_id; o=o.rename(columns={'pool':'opool','pun':'opun','n':'on'})
dec = dec.merge(o,on=['game','block','group_id'])
dec['pool_pc']=dec.pool/dec.n; dec['opool_pc']=dec.opool/dec.on.replace(0,np.nan)
print('switch rate', dec.switch.mean(), 'by round', dec.groupby('round_number').switch.mean().round(3).to_dict())
print(dec.groupby(pd.cut(dec.mypun,[-1,0,2,5,10,20,200]),observed=True).switch.agg(['mean','count']).round(3))
dec['dpc']=dec.pool_pc-dec.opool_pc
print(dec.groupby(pd.cut(dec.dpc,[-100,-10,-4,-1,1,4,10,100]),observed=True).switch.agg(['mean','count']).round(3))
dec['dpun']=dec.pun-dec.opun
print(dec.groupby(pd.cut(dec.dpun,[-30,-2,-0.5,0.5,2,30]),observed=True).switch.agg(['mean','count']).round(3))
import statsmodels.formula.api as smf
try:
    m = smf.logit('switch ~ mypun + dpc + dpun + myc + n', data=dec.dropna()).fit(disp=0); print(m.summary().tables[1])
except Exception as e: print(e)
# group sizes over time
s = df.groupby(['game','round_number','group_id']).size().unstack(fill_value=0)
print('size distribution at round 23', pd.Series(s.max(axis=1).xs(23,level=1)).value_counts().sort_index().to_dict())
