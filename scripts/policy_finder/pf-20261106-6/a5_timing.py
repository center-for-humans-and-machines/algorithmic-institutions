import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-6')
from load import load
import numpy as np, pandas as pd, statsmodels.formula.api as smf
df=load()
df['next_group']=df.groupby(['game','player_id']).group_id.shift(-1)
w=df.pivot_table(index=['game','player_id'],columns='round_number',values='punishment')
gp=df.groupby(['game','round_number','group_id']).punishment.mean().rename('gp').reset_index()
df=df.merge(gp,on=['game','round_number','group_id'])
gq=gp.copy(); gq['group_id']=1-gq.group_id; gq=gq.rename(columns={'gp':'op'})
df=df.merge(gq,on=['game','round_number','group_id'],how='left'); df['op']=df.op.fillna(0)
rows=[]
for d in [3,7,11,15,19]:
    x=df[df.round_number==d][['game','player_id','group_id','next_group']].copy()
    x['d']=d
    x['switch']=(x.next_group!=x.group_id).astype(int)
    for lag in range(4):
        y=df[df.round_number==d-lag][['game','player_id','punishment','gp','op','group_id','contribution']].rename(columns={'punishment':f'p{lag}','gp':f'gp{lag}','op':f'op{lag}','group_id':f'g{lag}','contribution':f'c{lag}'})
        x=x.merge(y,on=['game','player_id'])
    rows.append(x)
x=pd.concat(rows)
pg=df[['game','player_id','round_number','group_id']].copy(); pg['round_number']+=4
x['rprev']=x.g0*0
rows2=[]
for d in [3,7,11,15,19]:
    z=x[x.c0.notna()]
x=x.merge(df[['game','player_id','round_number','group_id']].rename(columns={'group_id':'gprev','round_number':'rp'}),how='left',left_on=['game','player_id'],right_on=['game','player_id'])
x=x[(x.rp==x.d-4)|((x.d==3)&(x.rp==0))]
x['joined']=((x.d>3)&(x.gprev!=x.group_id)).astype(int)
m=smf.logit('switch ~ p0+p1+p2+p3+gp0+gp1+gp2+gp3+op0+op1+op2+op3+joined',data=x).fit(disp=0)
print(m.summary2().tables[1].round(3))
print(x.groupby('joined').switch.mean())
# punishment receiving vs punished peers: own p vs group mean when own c high
x['hi']=(x.c0>=12)
print(x.groupby(['hi',pd.cut(x.gp0,[-1,0,1,3,30])],observed=True).switch.agg(['mean','size']))
