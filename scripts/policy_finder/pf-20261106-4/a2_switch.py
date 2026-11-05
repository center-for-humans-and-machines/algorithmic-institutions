import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-4')
from load import load
import numpy as np, pandas as pd
df, gs = load()
d = df[df.round_number.isin([3,7,11,15,19])].copy()
d['sw'] = (d.next_group != d.group_id).astype(float)
print('switch rate by decision round', d.groupby('round_number').sw.mean().round(3).to_dict())
# per-capita pool in own vs other group over the block (4 rounds)
blk = df.copy(); blk['block'] = blk.round_number//4
bg = blk.groupby(['game','block','group_id']).agg(cg=('common_good','mean'), n=('n','mean'), mp=('mp','mean'), mc=('mc','mean')).reset_index()
bg['pc'] = bg.cg/bg.n
d['block'] = d.round_number//4
pb = blk.groupby(['game','block','player_id']).agg(pblk=('punishment','sum'), cblk=('contribution','mean')).reset_index()
d = d.merge(pb, on=['game','block','player_id'])
own = bg.rename(columns={'cg':'cg_b','n':'n_b','mp':'mp_b','mc':'mc_b','pc':'pc_b'})
oth = bg.copy(); oth['group_id']=1-oth.group_id
oth = oth.rename(columns={'cg':'cgo_b','n':'no_b','mp':'mpo_b','mc':'mco_b','pc':'pco_b'})
d = d.merge(own,on=['game','block','group_id']).merge(oth,on=['game','block','group_id'],how='left')
d['dpc'] = d.pc_b - d.pco_b
d['dmp'] = d.mp_b - d.mpo_b
print('switch vs last-round own punishment')
print(d.groupby(pd.cut(d.punishment,[-1,0,2,5,10,30])).sw.agg(['mean','size']).round(3))
print('switch vs block punishment sum (own)')
print(d.groupby(pd.cut(d.pblk,[-1,0,4,10,20,200])).sw.agg(['mean','size']).round(3))
print('switch vs per-capita pool diff own-other (block)')
print(d.groupby(pd.cut(d.dpc,[-100,-10,-4,-1,1,4,10,100])).sw.agg(['mean','size']).round(3))
print('switch vs mean punishment diff own-other (block)')
print(d.groupby(pd.cut(d.dmp,[-30,-3,-1,-0.25,0.25,1,3,30])).sw.agg(['mean','size']).round(3))
print('switch vs own group size')
print(d.groupby('n').sw.agg(['mean','size']).round(3))
print('switch vs own contribution relative to group (block)')
d['rel'] = d.cblk - d.mc_b
print(d.groupby(pd.cut(d.rel,[-21,-5,-1,1,5,21])).sw.agg(['mean','size']).round(3))
import statsmodels.formula.api as smf
d2 = d.dropna(subset=['dpc','dmp'])
d2['pown']=d2.pblk/4
m = smf.logit('sw ~ dpc + dmp + pown + rel + n + round_number', d2).fit(disp=0)
print(m.summary().tables[1])
d.to_pickle('/tmp/sw.pkl')
