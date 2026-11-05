import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-1')
from load import load
import pandas as pd, numpy as np
df = load()
dec = df[df.round_number.isin([3,7,11,15,19])].copy()
dec['cbin']=pd.cut(dec.contribution,[-1,0,5,10,15,20])
dec['pbin']=pd.cut(dec.punishment,[-1,0,3,30])
print(dec.pivot_table(index='cbin',columns='pbin',values='switch',aggfunc='mean',observed=False).round(2))
print(dec.pivot_table(index='cbin',columns='pbin',values='switch',aggfunc='size',observed=False))
# manager style: per game-group, punishment rate (mean p per member-round), relative to other group's
gr = df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'), c=('contribution','sum'), p=('punishment','sum')).reset_index()
gr['pool']=1.6*gr.c-gr.p
full = pd.MultiIndex.from_product([gr.game.unique(), range(24), [0,1]], names=['game','round_number','group_id'])
gr = gr.set_index(['game','round_number','group_id']).reindex(full, fill_value=0).reset_index()
gg = gr.groupby(['game','group_id']).agg(pool=('pool','mean'), p=('p','sum'), mr=('n','sum'), c=('c','sum')).reset_index()
gg['prate']=gg.p/gg.mr.clip(lower=1)
gg['crate']=gg.c/gg.mr.clip(lower=1)
o = gg[['game','group_id','prate','pool','mr']].copy(); o['group_id']=1-o.group_id
gg = gg.merge(o, on=['game','group_id'], suffixes=('','_o'))
gg['rel']=gg.prate-gg.prate_o
gg['relbin']=pd.qcut(gg.rel,5)
print(gg.groupby('relbin',observed=False)[['prate','prate_o','crate','pool','pool_o','mr']].mean().round(2))
gg['pbin']=pd.qcut(gg.prate,5)
print(gg.groupby('pbin',observed=False)[['prate','crate','pool','mr']].mean().round(2))
# early rounds 0-3 (fixed size 4): punishment rate in block vs contribution in rounds 4..7 per stayer?
