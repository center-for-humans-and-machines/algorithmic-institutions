import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
df = pd.read_pickle('/tmp/df.pkl').sort_values(['game','player_id','round_number'])
df['c_next'] = df.groupby(['game','player_id']).contribution.shift(-1)
df['g_next'] = df.groupby(['game','player_id']).group_id.shift(-1)
df['dc'] = df.c_next - df.contribution
same = df[df.g_next == df.group_id]
same['pbin'] = pd.cut(same.punishment, [-1,0,2,5,10,30])
print('next-round change in contribution by punishment, split by own contribution level')
same['cl'] = pd.cut(same.contribution, [-1,5,10,15,20])
print(same.pivot_table(index='pbin', columns='cl', values='dc', aggfunc='mean').round(2))
print(same.pivot_table(index='pbin', columns='cl', values='dc', aggfunc='count'))
# switching: at decisions after rounds 3,7,11,15,19
sw = df[df.round_number.isin([3,7,11,15,19])].copy()
sw['switch'] = (sw.g_next != sw.group_id).astype(int)
print('switch rate', sw.switch.mean())
# own punishment cumulative in the block
df['block'] = df.round_number // 4
blk = df.groupby(['game','player_id','block']).agg(p=('punishment','sum'), c=('contribution','mean'), pay=('payoff','mean'), grp=('group_id','last')).reset_index()
gb = df.groupby(['game','block','group_id']).agg(gp=('punishment','mean'), gc=('contribution','mean'), gpay=('payoff','mean'), n=('contribution','size')).reset_index()
gb['n'] = gb.n/4
blk = blk.merge(gb, left_on=['game','block','grp'], right_on=['game','block','group_id'])
oth = gb.copy(); oth['group_id'] = 1-oth.group_id
blk = blk.merge(oth.rename(columns={'gp':'op','gc':'oc','gpay':'opay','n':'on'}), left_on=['game','block','grp'], right_on=['game','block','group_id'], how='left')
nxt = df[df.round_number.isin([4,8,12,16,20])][['game','player_id','block','group_id']].rename(columns={'group_id':'gnext'})
nxt['block'] -= 1
blk = blk.merge(nxt, on=['game','player_id','block'])
blk['switch'] = (blk.gnext != blk.grp).astype(int)
blk['pbin'] = pd.cut(blk.p, [-1,0,4,10,20,200])
print(blk.groupby('pbin').switch.agg(['mean','count']))
blk['paydiff'] = (blk.opay - blk.gpay)
blk['pdb'] = pd.cut(blk.paydiff, [-50,-4,-1,1,4,50])
print('switch by other group payoff - own group payoff'); print(blk.groupby('pdb').switch.agg(['mean','count']))
blk['gpd'] = pd.cut(blk.gp - blk.op, [-50,-2,-0.5,0.5,2,50])
print('switch by own-group mean pun - other'); print(blk.groupby('gpd').switch.agg(['mean','count']))
blk['cdb'] = pd.cut(blk.oc - blk.gc, [-21,-4,-1,1,4,21])
print('switch by other contrib - own group contrib'); print(blk.groupby('cdb').switch.agg(['mean','count']))
blk['nb'] = blk.n
print('switch by own group size'); print(blk.groupby('nb').switch.agg(['mean','count']))
print('switch by own contrib'); print(blk.groupby(pd.cut(blk.c,[-1,5,10,15,20])).switch.agg(['mean','count']))
