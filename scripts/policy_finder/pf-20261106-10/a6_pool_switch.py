# Switching vs own and other group's common_good in the decision round
import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-10')
import pandas as pd, numpy as np, statsmodels.formula.api as smf
from load import load
df=load()
gr=df.groupby(['game','round_number','group_id']).agg(cg=('common_good','first'),pm=('punishment','mean'),cm=('contribution','mean'),n=('player_id','count')).reset_index()
o=gr.copy(); o['group_id']=1-o.group_id; o.columns=['game','round_number','group_id','ocg','opm','ocm','on']
s=df[df.round_number.isin([3,7,11,15,19])].copy(); s['sw']=(s.next_group!=s.group_id).astype(float)
s=s.merge(gr,on=['game','round_number','group_id']).merge(o,on=['game','round_number','group_id'],how='left')
s=s.fillna({'ocg':0,'opm':0,'ocm':0,'on':0})
s['pc']=s.cg/s.n; s['opc']=np.where(s.on>0,s.ocg/s.on.clip(lower=1),0)
for f in ['sw ~ cg + ocg + punishment + pm + opm + n','sw ~ pc + opc + punishment + pm + opm + n','sw ~ cm + ocm + punishment + pm + opm + n + contribution']:
    m=smf.logit(f,data=s).fit(disp=0); print(f); print(pd.DataFrame({'b':m.params,'se':m.bse}).round(3).T)
# who switches in: contribution of switchers vs stayers in destination
s['next_c']=s['next_c']
print('next c of switchers',s[s.sw==1].next_c.mean(),'stayers',s[s.sw==0].next_c.mean())
