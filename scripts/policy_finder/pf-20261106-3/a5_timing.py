import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-3')
from load import load
import numpy as np, pandas as pd, statsmodels.formula.api as smf
df = load().sort_values(['game','player_id','round_number'])
df['next_group'] = df.groupby(['game','player_id']).group_id.shift(-1)
df['block']=df.round_number//4; df['pos']=df.round_number%4
gm = df.groupby(['game','round_number','group_id']).agg(gp=('punishment','mean'),n=('player_id','size'),cg=('common_good','first')).reset_index()
gm['cgpc']=gm.cg/gm.n
df = df.merge(gm,on=['game','round_number','group_id'])
o = gm.copy(); o['group_id']=1-o.group_id
df = df.merge(o[['game','round_number','group_id','gp','cgpc']],on=['game','round_number','group_id'],how='left',suffixes=('','_o'))
w = df[df.block<5].pivot_table(index=['game','player_id','block'],columns='pos',values=['punishment','cgpc','gp_o','cgpc_o'])
w.columns=[f'{a}{b}' for a,b in w.columns]
w=w.reset_index()
dec = df[df.pos==3][['game','player_id','block','group_id','next_group','round_number']]
dec['switch']=(dec.next_group!=dec.group_id).astype(int)
d=dec.merge(w,on=['game','player_id','block']).dropna()
m=smf.logit('switch ~ punishment0+punishment1+punishment2+punishment3+cgpc0+cgpc1+cgpc2+cgpc3+cgpc_o3+gp_o3+gp_o0',d).fit(disp=0)
print(m.summary().tables[1])
