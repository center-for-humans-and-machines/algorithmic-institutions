import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-6')
from load import load
import numpy as np, pandas as pd
df=load()
df['next_group']=df.groupby(['game','player_id']).group_id.shift(-1)
dec=df[df.round_number.isin([3,7,11,15,19])].copy()
dec['switch']=(dec.next_group!=dec.group_id).astype(int)
print('switch rate', dec.switch.mean(), dec.groupby('round_number').switch.mean().round(3).to_dict())
# per player stats over the block of 4 rounds
df['block']=df.round_number//4
blk=df.groupby(['game','player_id','block']).agg(c=('contribution','mean'),p=('punishment','mean'),pay=('payoff','mean'),g=('group_id','last'),pn=('punishment',lambda x:(x>0).mean())).reset_index()
grp=df.groupby(['game','block','group_id']).agg(gp=('punishment','mean'),gc=('contribution','mean'),gpay=('payoff','mean'),gn=('player_id','nunique')).reset_index()
blk=blk.merge(grp,left_on=['game','block','g'],right_on=['game','block','group_id'])
oth=grp.rename(columns={'group_id':'og','gp':'op','gc':'oc','gpay':'opay','gn':'on'})
blk['og']=1-blk.g
blk=blk.merge(oth,on=['game','block','og'],how='left')
d=dec[['game','player_id','round_number','switch']].copy(); d['block']=d.round_number//4
blk=blk.merge(d,on=['game','player_id','block'])
blk['dpay']=blk.gpay-blk.opay; blk['dp']=blk.gp-blk.op; blk['dc']=blk.gc-blk.oc
print(blk[['switch','p','pn','pay','c','gp','gc','gpay','gn','dpay','dp','dc']].corr()['switch'].round(3))
blk['pbin']=pd.cut(blk.p,[-0.1,0,1,3,6,31])
print(blk.groupby('pbin').switch.agg(['mean','size']))
blk['dpbin']=pd.cut(blk.dp,[-31,-3,-1,-0.1,0.1,1,3,31])
print(blk.groupby('dpbin').switch.agg(['mean','size']))
blk['dpaybin']=pd.qcut(blk.dpay,6)
print(blk.groupby('dpaybin').switch.agg(['mean','size']))
blk['cbin']=pd.cut(blk.c,[-1,2,6,10,14,18,20])
print(blk.groupby('cbin').switch.agg(['mean','size']))
import statsmodels.formula.api as smf
try:
    m=smf.logit('switch ~ p + dp + dc + c + gn',data=blk.dropna()).fit(disp=0); print(m.summary2().tables[1].round(3))
except Exception as e: print(e)
blk.to_pickle('/tmp/blk.pkl')
