import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261112-3')
from human_load import load
import pandas as pd, numpy as np
d=load()
print(d.game.nunique(),'games', len(d))
g=d.groupby('player_id' if False else ['game','player_id'])
d['next_group']=d.groupby(['game','player_id']).group_id.shift(-1)
dec=d[d.round_number.isin([3,7,11,15,19])].copy()
dec['switch']=(dec.next_group!=dec.group_id).astype(float)
# block features: punishment received over last 4 rounds, own vs other group's pool
blk=d[d.round_number.isin(range(0,24))].copy()
blk['block']=blk.round_number//4
pp=blk.groupby(['game','player_id','block']).agg(p4=('punishment','mean'),c4=('contribution','mean'),pay4=('payoff','mean')).reset_index()
dec['block']=dec.round_number//4
dec=dec.merge(pp,on=['game','player_id','block'])
# group sizes & group mean payoffs
gs=d.groupby(['game','round_number','group_id']).agg(n=('player_id','size'),gpay=('payoff','mean'),gp=('punishment','mean'),gc=('contribution','mean')).reset_index()
dec=dec.merge(gs,on=['game','round_number','group_id'])
oth=gs.rename(columns={'group_id':'og','n':'on','gpay':'ogpay','gp':'ogp','gc':'ogc'})
oth['group_id']=1-oth.og
dec=dec.merge(oth.drop(columns='og'),on=['game','round_number','group_id'],how='left')
print('overall switch rate',dec.switch.mean().round(3))
print(dec.groupby('round_number').switch.mean().round(3))
dec['pbin']=pd.cut(dec.p4,[-1,0,1,3,6,30])
print(dec.groupby('pbin').switch.agg(['mean','size']).round(3))
dec['gpbin']=pd.cut(dec.gp-dec.ogp.fillna(0),[-30,-2,-0.5,0.5,2,30])
print('group punishment minus other group',dec.groupby('gpbin').switch.agg(['mean','size']).round(3))
dec['paybin']=pd.cut(dec.gpay-dec.ogpay,[-50,-5,-1,1,5,50])
print('payoff diff (last round)',dec.groupby('paybin').switch.agg(['mean','size']).round(3))
dec['p_now']=pd.cut(dec.punishment,[-1,0,2,5,30])
print('punishment at decision round',dec.groupby('p_now').switch.agg(['mean','size']).round(3))
import statsmodels.formula.api as smf
dec['dgp']=dec.gp-dec.ogp; dec['dgc']=dec.gc-dec.ogc; dec['dn']=dec.n-dec.on
m=smf.logit('switch ~ p4 + punishment + dgp + dgc + dn + c4', data=dec.dropna()).fit(disp=0)
print(m.summary2().tables[1].round(3))
