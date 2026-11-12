import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d=d[d.run.str.contains('ah_vs_zero|zero_vs_ah')].copy()
d['mgr']=np.where(d.run.str.contains('ah_vs_zero'), np.where(d.agent_group==0,'ah','zero'), np.where(d.agent_group==0,'zero','ah'))
d=d.sort_values(['run','episode','participant_code','round_number'])
k=['run','episode','participant_code']
d['c_next']=d.groupby(k).contribution.shift(-1)
d['g_next']=d.groupby(k).agent_group.shift(-1)
x=d[d.c_next.notna()&(d.g_next==d.agent_group)&~d.round_number.isin([3,7,11,15,19])&d.contribution_valid.astype(bool)].copy()
x['dc']=x.c_next-x.contribution
x['cbin']=pd.cut(x.contribution,[-1,4,9,14,19,20]); x['pbin']=pd.cut(x.punishment,[-1,0,2,5,10,30])
print(x[x.mgr=='ah'].pivot_table(index='cbin',columns='pbin',values='dc',aggfunc='mean').round(2))
print(x[x.mgr=='zero'].groupby('cbin').dc.mean().round(2))
# switching
d['blk']=d.round_number//4
pb=d.groupby(k+['blk']).punishment.mean().rename('p4').reset_index()
dec=d[d.round_number.isin([3,7,11,15,19])].merge(pb,on=k+['blk'])
dec['sw']=(dec.g_next!=dec.agent_group).astype(float)
print('switch rate from ah vs zero groups', dec.groupby('mgr').sw.mean().round(3).to_dict())
dec['pb']=pd.cut(dec.p4,[-1,0,1,3,6,30])
print(dec[dec.mgr=='ah'].groupby('pb').sw.agg(['mean','size']).round(3))
dec['cb']=pd.cut(dec.contribution,[-1,4,9,14,20])
print(dec.pivot_table(index='cb',columns='mgr',values='sw').round(3))
print(dec.groupby(['mgr','round_number']).sw.mean().unstack().round(3))
