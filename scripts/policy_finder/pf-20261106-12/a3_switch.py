import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-12')
from load import load
import numpy as np, pandas as pd
d, gs = load()
d = d.sort_values(['game','player_id','round_number'])
# at decision rounds 3,7,..: switch = grp_next != group_id
dec = d[d.round_number.isin([3,7,11,15,19])].copy()
dec['sw'] = (dec.grp_next != dec.group_id).astype(float)
print('switch rate overall', dec.sw.mean().round(3), 'by round', dec.groupby('round_number').sw.mean().round(3).to_dict())
# punishment in the block (4 rounds) up to decision
d['blk'] = d.round_number//4
bp = d.groupby(['game','player_id','blk']).agg(pblk=('punishment','sum'), cblk=('contribution','mean')).reset_index()
dec['blk'] = dec.round_number//4
dec = dec.merge(bp, on=['game','player_id','blk'])
# other group's pool this round
og = gs.copy(); og['group_id'] = 1-og.group_id
og = og.rename(columns={'pool':'opool','n':'on','pmean':'opmean','cmean':'ocmean'})[['game','round_number','group_id','opool','on','opmean','ocmean']]
dec = dec.merge(og, on=['game','round_number','group_id'], how='left')
dec['on'] = dec.on.fillna(0); dec['opool']=dec.opool.fillna(0)
dec['pb'] = pd.cut(dec.pblk, [-1,0,2,5,10,20,200])
print(dec.groupby('pb', observed=True).sw.agg(['mean','size']).round(3))
dec['ppr'] = dec.pool/dec.n; dec['oppr'] = dec.opool/dec.on.replace(0,np.nan)
dec['diff'] = (dec.oppr - dec.ppr)
dec['db'] = pd.cut(dec['diff'], [-100,-10,-3,3,10,100])
print('switch by (other per-capita pool - own per-capita pool):')
print(dec.groupby('db', observed=True).sw.agg(['mean','size']).round(3))
print('switch by own per-capita pool:')
print(dec.groupby(pd.cut(dec.ppr,[-50,5,10,15,20,25,30,60]), observed=True).sw.agg(['mean','size']).round(3))
print('switch by last-round punishment:')
print(dec.groupby(pd.cut(dec.punishment,[-1,0,2,5,10,30]), observed=True).sw.agg(['mean','size']).round(3))
print('switch by own block mean contribution:')
print(dec.groupby(pd.cut(dec.cblk,[-1,3,7,11,15,20]), observed=True).sw.agg(['mean','size']).round(3))
print('switch by group size:'); print(dec.groupby('n').sw.agg(['mean','size']).round(3))
# logistic regression
from numpy.linalg import lstsq
import statsmodels.api as sm
X = dec[['pblk','ppr','oppr','cblk','n']].fillna(0).copy(); X['o_empty']=(dec.on==0).astype(float)
X['ppr']=X.ppr/10; X['oppr']=X.oppr/10
m = sm.Logit(dec.sw, sm.add_constant(X)).fit(disp=0); print(m.summary2().tables[1].round(3))
