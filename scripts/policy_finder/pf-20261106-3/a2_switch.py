import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-3')
from load import load
import numpy as np, pandas as pd
df = load().sort_values(['game','player_id','round_number'])
df['next_group'] = df.groupby(['game','player_id']).group_id.shift(-1)
dec = df[df.round_number.isin([3,7,11,15,19])].copy()
dec['switch'] = (dec.next_group != dec.group_id).astype(int)
print('switch rate', dec.switch.mean(), 'by round', dec.groupby('round_number').switch.mean().round(3).to_dict())
# group-level stats over the block (last 4 rounds) for own and other group
df['block'] = df.round_number//4
gb = df.groupby(['game','block','round_number','group_id']).agg(n=('player_id','size'),c=('contribution','mean'),p=('punishment','mean'),cg=('common_good','first')).reset_index()
gb['cgpc']=gb.cg/gb.n
bl = gb.groupby(['game','block','group_id']).agg(c=('c','mean'),p=('p','mean'),cgpc=('cgpc','mean'),n=('n','mean')).reset_index()
dec['block']=dec.round_number//4
dec = dec.merge(bl, on=['game','block','group_id'])
oth = bl.copy(); oth['group_id']=1-oth.group_id
dec = dec.merge(oth, on=['game','block','group_id'], suffixes=('','_o'), how='left')
# own punishment in block
own = df.groupby(['game','player_id','block']).agg(myp=('punishment','mean'),myc=('contribution','mean')).reset_index()
dec = dec.merge(own,on=['game','player_id','block'])
dec['dcg'] = dec.cgpc - dec.cgpc_o
dec['dp'] = dec.p - dec.p_o
for col in ['dcg','dp','myp','p','cgpc','myc']:
    q = pd.qcut(dec[col].rank(method='first'),5,labels=False)
    print(col, dec.groupby(q).agg(x=(col,'mean'),sw=('switch','mean'),n=('switch','size')).round(2).to_dict('list'))
import statsmodels.formula.api as smf
d = dec.dropna(subset=['cgpc_o'])
m = smf.logit('switch ~ cgpc + cgpc_o + p + p_o + myp + myc + n + round_number', d).fit(disp=0)
print(m.summary().tables[1])
