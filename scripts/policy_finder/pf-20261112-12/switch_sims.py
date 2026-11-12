"""What drives switching in the reference sims (artificial humans)."""
import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d=d.sort_values(['run','episode','participant_code','round_number'])
d['blk']=d.round_number//4
d['pc_cg']=d.common_good
# block-level per player
b=d.groupby(['run','episode','participant_code','blk']).agg(g=('agent_group','first'),p=('punishment','mean'),p_last=('punishment','last'),c=('contribution','mean'),cg=('common_good','mean'),pay=('payoff','mean')).reset_index()
nxt=d.groupby(['run','episode','participant_code','blk']).agg(g=('agent_group','first')).reset_index()
nxt['blk']-=1; nxt=nxt.rename(columns={'g':'gnext'})
b=b.merge(nxt,on=['run','episode','participant_code','blk'])
b['sw']=(b.g!=b.gnext).astype(int)
# group-level block features
gb=d.groupby(['run','episode','blk','agent_group']).agg(gcg=('common_good','mean'),gp=('punishment','mean'),gpay=('payoff','mean'),n=('payoff','size'),gc=('contribution','mean')).reset_index()
gb['n']/=4
b=b.merge(gb.rename(columns={'agent_group':'g'}),on=['run','episode','blk','g'])
o=gb.rename(columns={'agent_group':'og','gcg':'ocg','gp':'op','gpay':'opay','n':'on','gc':'oc'}); o['g']=1-o.og
b=b.merge(o.drop(columns='og'),on=['run','episode','blk','g'],how='left').fillna({'ocg':0,'op':0,'opay':0,'on':0,'oc':0})
print('overall switch rate', b.sw.mean())
print(b.groupby('run').sw.mean())
b['pbin']=pd.cut(b.p,[-0.1,0,1,3,6,10,30])
print(b.groupby('pbin').sw.agg(['mean','size']))
b['dpay']=b.pay-b.opay
b['dpbin']=pd.qcut(b.gpay-b.opay,8,duplicates='drop')
print(b.groupby('dpbin').sw.agg(['mean','size']))
b['dp_grp']=pd.cut(b.gp-b.op,[-30,-3,-1,-0.25,0.25,1,3,30])
print(b.groupby('dp_grp').sw.agg(['mean','size']))
# logistic regression
import statsmodels.formula.api as smf
b['ownpun']=b.p-b.gp
m=smf.logit('sw ~ p + gp + op + gpay + opay + c + gc + oc + n + on',data=b.sample(100000,random_state=0)).fit(disp=0)
print(m.summary2().tables[1].round(3))
b.drop(columns=["pbin","dpbin","dp_grp"]).to_parquet("/tmp/pf12_blocks.parquet")
