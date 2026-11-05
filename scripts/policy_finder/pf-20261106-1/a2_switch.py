import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-1')
from load import load
import pandas as pd, numpy as np
df = load()
dec = df[df.round_number.isin([3,7,11,15,19])].copy()
# per-capita payoffs of groups at the decision round, and over the block
gr = df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'), cg=('common_good','first'), pm=('punishment','mean'), cm=('contribution','mean')).reset_index()
gr['pc'] = gr.cg/gr.n
dec = dec.merge(gr, on=['game','round_number','group_id'])
oth = gr.copy(); oth['group_id']=1-oth.group_id
oth = oth.rename(columns={'n':'n_o','cg':'cg_o','pm':'pm_o','cm':'cm_o','pc':'pc_o'})
dec = dec.merge(oth, on=['game','round_number','group_id'], how='left')
dec['n_o']=dec.n_o.fillna(0)
# block-level own punishment (last 4 rounds)
df['blk'] = df.round_number//4
bp = df.groupby(['game','player_id','blk']).agg(pblk=('punishment','sum'), cblk=('contribution','mean')).reset_index()
dec['blk']=dec.round_number//4
dec = dec.merge(bp, on=['game','player_id','blk'])
print('switch rate', dec.switch.mean(), 'by round', dec.groupby('round_number').switch.mean().round(3).to_dict())
dec['pbin']=pd.cut(dec.punishment,[-1,0,2,5,10,30])
print(dec.groupby('pbin').switch.agg(['mean','size']))
dec['pblkbin']=pd.cut(dec.pblk,[-1,0,4,10,20,40,200])
print(dec.groupby('pblkbin').switch.agg(['mean','size']))
dec['dpc']=dec.pc - dec.pc_o
dec['dpcbin']=pd.cut(dec.dpc,[-100,-10,-4,-1,1,4,10,100])
print(dec[dec.n_o>0].groupby('dpcbin').switch.agg(['mean','size']))
dec['dpm']=dec.pm - dec.pm_o
dec['dpmbin']=pd.cut(dec.dpm,[-40,-3,-1,-0.01,0.01,1,3,40])
print('own grp mean pun minus other', dec[dec.n_o>0].groupby('dpmbin').switch.agg(['mean','size']))
print('by own group size', dec.groupby('n').switch.agg(['mean','size']))
# logistic regression
import statsmodels.formula.api as smf
d = dec[dec.n_o>0].dropna(subset=['switch'])
d['pcm']=d.pc - d.pc_o
m = smf.logit('switch ~ punishment + pblk + pcm + pm + pm_o + n + contribution + C(round_number)', d).fit(disp=0)
print(m.summary().tables[1])
