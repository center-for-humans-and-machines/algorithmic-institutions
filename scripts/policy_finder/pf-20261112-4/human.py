import pandas as pd, numpy as np
import statsmodels.formula.api as smf
d=pd.read_csv('experiments/2group_8agent_50ep.csv').drop_duplicates(['session','episode','round_number','participant_code'])
d['c']=d.contribution
d=d.sort_values(['session','episode','participant_code','round_number'])
g=d.groupby(['session','episode','participant_code'])
d['ng']=g.group_id.shift(-1); d['sw']=(d.ng!=d.group_id).astype(float); d['nc']=g.c.shift(-1)
for k in [1,2,3]: d[f'p{k}']=g.punishment.shift(k)
pool=d.groupby(['session','episode','round_number','group_id']).agg(pool=('common_good','first'),n=('c','size'),gc=('c','mean'),gp=('punishment','mean')).reset_index()
op=pool[['session','episode','round_number','group_id','pool','n']].copy(); op['group_id']=1-op.group_id; op=op.rename(columns={'pool':'opool','n':'on'})
d=d.merge(pool,on=['session','episode','round_number','group_id']).merge(op,on=['session','episode','round_number','group_id'],how='left')
d['opool']=d.opool.fillna(0); d['on']=d.on.fillna(0)
print('games',d.groupby(['session','episode']).ngroups)
# managers per game: group means
gm=pool.groupby(['session','episode','group_id']).agg(pool=('pool','mean'),p=('gp','mean'),c=('gc','mean'),n=('n','mean')).reset_index()
print(gm.describe().round(2))
print('corr of game-group mean pool with mean punishment', gm[['pool','p','c','n']].corr().round(2))
dec=d[d.round_number.isin([3,7,11,15,19])].copy()
dec['dpool']=(dec.pool-dec.opool)/10
print('switch rate',dec.sw.mean(), dec.groupby('round_number').sw.mean().round(3).to_dict())
m=smf.logit('sw ~ punishment + p1 + p2 + p3 + c + dpool + C(round_number)',data=dec.dropna(subset=['p3'])).fit(disp=0)
print(m.summary().tables[1])
# contribution response
y=d[(d.round_number<23)&(d.sw==0)].copy()
y['dc']=y.nc-y.c
m2=smf.ols('nc ~ c + punishment + I(punishment*(c<10)) + gc + round_number',data=y).fit(cov_type='cluster',cov_kwds={'groups':y.session+y.episode.astype(str)})
print(m2.summary().tables[1])
y['pb']=pd.cut(y.punishment,[-1,0,2,5,10,30]); y['cb']=pd.cut(y.c,[-1,4,9,14,20])
print(y.groupby(['cb','pb'],observed=True).dc.agg(['mean','size']).unstack(1).round(2))
# what did human managers do: punishment by contribution and gap to group mean
d['gap']=d.gc-d.c
d['gb']=pd.cut(d.c,[-1,0,4,9,14,19,20])
print(d.groupby('gb',observed=True).punishment.agg(['mean',lambda s:(s>0).mean(),'size']).round(2))
print(d.groupby('round_number').punishment.mean().round(2).to_dict())
