import pandas as pd, numpy as np
d=pd.read_csv('experiments/2group_8agent_50ep.csv')
d=d.drop_duplicates(['session','episode','round_number','participant_code'])
d['game']=d.session.astype(str)+'_'+d.episode.astype(str)
print(d.shape, d.game.nunique())
d=d.sort_values(['game','player_id','round_number'])
d['next_group']=d.groupby(['game','player_id']).group_id.shift(-1)
d['blk']=d.round_number//4
dec=d[d.round_number.isin([3,7,11,15,19])].copy()
dec['sw']=(dec.next_group!=dec.group_id).astype(float)
print('switch rate',dec.sw.mean())
# window features: last 4 rounds
d['blk']=d.round_number//4
g=d.groupby(['game','blk','group_id']).agg(gp=('punishment','mean'),gc=('contribution','mean'),gn=('player_id','nunique'),gcg=('common_good','mean')).reset_index()
own=d.groupby(['game','blk','player_id']).agg(op=('punishment','mean'),oc=('contribution','mean'),grp=('group_id','last'),pay=('payoff','mean')).reset_index()
x=own.merge(g,left_on=['game','blk','grp'],right_on=['game','blk','group_id'])
g2=g.rename(columns={'group_id':'oth','gp':'xp','gc':'xc','gn':'xn','gcg':'xcg'})
x['oth']=1-x.grp
x=x.merge(g2,on=['game','blk','oth'],how='left')
dec=dec.merge(x[['game','blk','player_id','op','oc','gp','gc','gn','gcg','xp','xc','xn','xcg','pay']],on=['game','blk','player_id'])
dec['dp']=dec.gp-dec.xp
dec['dcg']=dec.gcg-dec.xcg
for col,bins in [('op',[-.1,0,1,3,6,31]),('dp',[-31,-2,-.5,.5,2,31]),('dcg',[-50,-5,-1,1,5,50]),('gn',[0,2,3,4,5,6,8])]:
    print(dec.groupby(pd.cut(dec[col],bins)).sw.agg(['mean','size']).round(3))
import statsmodels.formula.api as smf
try:
    m=smf.logit('sw ~ op + dp + dcg + gn + oc',data=dec.dropna()).fit(disp=0); print(m.summary().tables[1])
except Exception as e: print(e)
