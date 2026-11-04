"""Reference sims: switching and contribution response to punishment under ah."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d.sort_values(['run','episode','participant_code','round_number'])
g = d.groupby(['run','episode','participant_code'])
d['next_group'] = g.agent_group.shift(-1); d['next_c'] = g.contribution.shift(-1)
d['switched'] = (d.next_group != d.agent_group).astype(float)
d['block'] = d.round_number // 4
gs = d.groupby(['run','episode','round_number','agent_group']).agg(gc=('contribution','mean'), gp=('punishment','mean'), n=('contribution','size')).reset_index()
d = d.merge(gs, on=['run','episode','round_number','agent_group'])
bs = d.groupby(['run','episode','participant_code','block']).agg(pun=('punishment','sum'), con=('contribution','mean'), gpb=('gp','mean'), gcb=('gc','mean'), pun_last=('punishment','last')).reset_index()
dec = d[d.round_number.isin([3,7,11,15,19])].merge(bs, on=['run','episode','participant_code','block'])
dec['mgr'] = np.where(dec.run.str.contains('ah_vs_ah'), 'ah', np.where((dec.agent_group==0) == dec.run.str.contains('by ah_vs'), 'ah', 'zero'))
print('switch rate by manager, decision'); print(dec.groupby(['mgr','round_number']).switched.mean().unstack().round(3))
a = dec[dec.mgr=='ah']
print('ah groups: switch by own block punishment'); print(a.groupby(pd.cut(a.pun,[-1,0,4,10,20,40,500])).switched.agg(['mean','size']).round(3))
print('ah groups: switch by group mean punishment/round'); print(a.groupby(pd.cut(a.gpb,[-1,0,1,2.5,5,100])).switched.agg(['mean','size']).round(3))
print('ah groups: switch by punishment in last round of block only'); print(a.groupby(pd.cut(a.pun_last,[-1,0,2,5,10,30])).switched.agg(['mean','size']).round(3))
z = dec[dec.mgr=='zero']
print('zero groups: switch by own block contribution'); print(z.groupby(pd.cut(z.con,[-1,5,10,15,20])).switched.agg(['mean','size']).round(3))
print('ah groups: switch by own contrib x own punishment'); print(a.groupby([pd.cut(a.con,[-1,5,10,15,20]), pd.cut(a.pun,[-1,0,4,10,20,500])]).switched.mean().unstack().round(2))
# contribution response in ah groups
x = d[(d.round_number<23)&(d.switched==0)].copy()
x['mgr'] = np.where(x.run.str.contains('ah_vs_ah'), 'ah', np.where((x.agent_group==0) == x.run.str.contains('by ah_vs'), 'ah', 'zero'))
x['dc'] = x.next_c - x.contribution
x = x[x.contribution_valid]
print('next-round dc, by own c (rows) and punishment (cols); zero groups in column p=0 zero')
r = x[x.mgr=='ah'].groupby([pd.cut(x.contribution,[-1,0,5,10,15,19,20]), pd.cut(x.punishment,[-1,0,2,5,10,30])]).dc.mean().unstack().round(2)
r['zero_mgr'] = x[x.mgr=='zero'].groupby(pd.cut(x.contribution,[-1,0,5,10,15,19,20])).dc.mean().round(2)
print(r)
print(x[x.mgr=='ah'].groupby([pd.cut(x.contribution,[-1,0,5,10,15,19,20]), pd.cut(x.punishment,[-1,0,2,5,10,30])]).dc.size().unstack())
