from load import load
import pandas as pd, numpy as np
df=load().sort_values(['game','player_id','round_number'])
g=df.groupby(['game','player_id'])
df['next_group']=g.group_id.shift(-1)
grp=df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'),cmean=('contribution','mean'),psum=('punishment','sum'),pmean=('punishment','mean'),cg=('common_good','first')).reset_index()
grp['pc']=grp.cg/grp.n
df=df.merge(grp,on=['game','round_number','group_id'])
oth=grp.rename(columns={c:'o_'+c for c in ['n','cmean','psum','pmean','cg','pc']}); oth['group_id']=1-oth.group_id
df=df.merge(oth,on=['game','round_number','group_id'],how='left')
for c in ['o_n','o_cmean','o_pc','o_pmean']: df[c]=df[c].fillna(0)
dec=df[df.round_number.isin([3,7,11,15,19])&df.next_group.notna()].copy()
dec['sw']=(dec.next_group!=dec.group_id).astype(int)
# own punishment over the last 4 rounds (block)
df['blk']=df.round_number//4
bp=df.groupby(['game','player_id','blk']).punishment.mean().rename('blkp').reset_index()
bg=df.groupby(['game','player_id','blk']).agg(blk_pc=('pc','mean'),blk_opc=('o_pc','mean'),blk_pm=('pmean','mean'),blk_opm=('o_pmean','mean')).reset_index()
dec['blk']=dec.round_number//4
dec=dec.merge(bp,on=['game','player_id','blk']).merge(bg,on=['game','player_id','blk'])
print('switch rate',dec.sw.mean().round(3), 'by round', dec.groupby('round_number').sw.mean().round(3).to_dict())
dec['dpc']=dec.pc-dec.o_pc
import statsmodels.formula.api as smf
r=smf.logit('sw ~ pc + o_pc + punishment + blkp + contribution + n',dec).fit(disp=0); print(r.summary2().tables[1].round(3))
r=smf.logit('sw ~ blk_pc + blk_opc + blkp + blk_pm + blk_opm + contribution+n',dec).fit(disp=0); print(r.summary2().tables[1].round(3))
dec['pb']=pd.cut(dec.punishment,[-1,0,3,8,30])
print(dec.groupby('pb').sw.agg(['mean','size']))
dec['bpb']=pd.cut(dec.blkp,[-1,0,1,3,8,30])
print(dec.groupby('bpb').sw.agg(['mean','size']))
dec['db']=pd.qcut(dec.dpc,5)
print(dec.groupby('db').sw.agg(['mean','size']))
# switch by own contribution and punished-or-not
dec['cb']=pd.cut(dec.contribution,[-1,5,10,15,20])
print(dec.pivot_table(index='cb',columns=dec.blkp>0,values='sw',aggfunc=['mean','size']))
