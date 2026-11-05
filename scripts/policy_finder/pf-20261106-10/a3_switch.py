# What drives switching? Switch decisions after rounds 3,7,11,15,19.
import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-10')
import pandas as pd, numpy as np
from load import load
df=load()
d=df[df.round_number.isin([3,7,11,15,19])].copy()
d['sw']=(d.next_group!=d.group_id).astype(float)
print('switch rate', d.sw.mean(), d.groupby('round_number').sw.mean().round(3).to_dict())
# group-level stats over the block of 4 rounds
df['block']=df.round_number//4
gb=df.groupby(['game','block','group_id']).agg(pool=('common_good','mean'),pun=('punishment','mean'),c=('contribution','mean')).reset_index()
d['block']=d.round_number//4
d=d.merge(gb,on=['game','block','group_id'])
other=gb.copy(); other['group_id']=1-other.group_id
d=d.merge(other.rename(columns={'pool':'opool','pun':'opun','c':'oc'}),on=['game','block','group_id'],how='left')
# own block punishment
pb=df.groupby(['game','block','player_id']).punishment.sum().rename('own_pun_block').reset_index()
d=d.merge(pb,on=['game','block','player_id'])
d['ownpb']=pd.cut(d.own_pun_block,[-1,0,4,10,200])
print(d.groupby('ownpb',observed=True).sw.agg(['mean','count']).round(3))
d['pcdiff']=(d.pool/d['size'])
d['pcd']=pd.cut(d.c-d.oc.fillna(d.c),[-21,-4,-1,1,4,21])
print('by own-group minus other-group mean contribution'); print(d.groupby('pcd',observed=True).sw.agg(['mean','count']).round(3))
d['gpun']=pd.cut(d.pun-d.opun.fillna(d.pun),[-31,-2,-0.5,0.5,2,31])
print('by own minus other mean punishment'); print(d.groupby('gpun',observed=True).sw.agg(['mean','count']).round(3))
print(d.groupby('size').sw.agg(['mean','count']).round(3))
import statsmodels.formula.api as smf
d['dc']=d.c-d.oc; d['dp']=d.pun-d.opun
m=smf.logit('sw ~ own_pun_block + dc + dp + size + contribution', data=d.dropna(subset=['dc','dp'])).fit(disp=0)
print(m.summary().tables[1])
