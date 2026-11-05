import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from load import load
import pandas as pd, numpy as np
df=load()
df['next_group']=df.groupby(['game','player_id']).group_id.shift(-1)
df['block']=df.round_number//4
# block level stats per player: own mean punishment, own mean payoff, own group mean c, other group's mean punishment/payoff
blk=df.groupby(['game','block','round_number','group_id']).agg(gp=('punishment','mean'),gc=('contribution','mean'),gpay=('payoff','mean'),gn=('player_id','size')).reset_index()
d=df.merge(blk,on=['game','block','round_number','group_id'])
oth=blk.copy(); oth['group_id']=1-oth.group_id; oth=oth.rename(columns={'gp':'op','gc':'oc','gpay':'opay','gn':'on'})
d=d.merge(oth,on=['game','block','round_number','group_id'],how='left')
dec=d[d.round_number%4==3][d.round_number<23].copy()
B=d[d.round_number//4==d.block].groupby(['game','player_id','block']).agg(myp=('punishment','mean'),myc=('contribution','mean'),mypay=('payoff','mean'),gp=('gp','mean'),op=('op','mean'),gpay=('gpay','mean'),opay=('opay','mean'),gn=('gn','mean'),on=('on','mean')).reset_index()
dec=dec[['game','player_id','block','group_id','next_group']].merge(B,on=['game','player_id','block'])
dec['switch']=(dec.next_group!=dec.group_id).astype(float)
dec=dec.dropna(subset=['next_group'])
print('switch rate',dec.switch.mean().round(3), 'n',len(dec))
print(dec.groupby('block').switch.mean().round(3).to_dict())
dec['dpay']=dec.gpay-dec.opay.fillna(dec.gpay)
dec['dp']=dec.gp-dec.op.fillna(0)
for col,b in [('myp',[-1,0,1,3,6,31]),('dpay',[-100,-10,-4,-1,1,4,10,100]),('dp',[-31,-3,-1,-0.25,0.25,1,3,31]),('mypay',[-100,15,20,25,30,100]),('gn',[0,2,3,4,5,6,8])]:
    print(col); print(dec.groupby(pd.cut(dec[col],b),observed=True).switch.agg(['mean','size']).round(3).to_string())
# myp relative to own contribution
dec['myc_bin']=pd.cut(dec.myc,[-1,5,10,15,20])
print(dec.pivot_table(index='myc_bin',columns=pd.cut(dec.myp,[-1,0,2,5,31]),values='switch',aggfunc='mean',observed=True).round(2))
import statsmodels.formula.api as smf
m=smf.logit('switch ~ myp + dpay + gn + myc', data=dec.dropna()).fit(disp=0); print(m.summary().tables[1])
