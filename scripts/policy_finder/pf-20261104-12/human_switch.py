"""Human data: who switches, and how contributions respond to punishment."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session','episode','round_number','participant_code'])
d = d.sort_values(['session','episode','participant_code','round_number'])
print(len(d), d.groupby(['session','episode']).ngroups, 'games')
g = d.groupby(['session','episode','participant_code'])
d['next_group'] = g.group_id.shift(-1)
d['next_c'] = g.contribution.shift(-1)
d['switched'] = (d.next_group != d.group_id).astype(float)
# group stats
gs = d.groupby(['session','episode','round_number','group_id']).agg(gc=('contribution','mean'), gp=('punishment','mean'), n=('contribution','size')).reset_index()
d = d.merge(gs, on=['session','episode','round_number','group_id'])
# block-level: punishment received in last 4 rounds
d['pblock'] = g.punishment.transform(lambda s: s.rolling(4, min_periods=1).sum()) if False else 0
dec = d[d.round_number.isin([3,7,11,15,19])].copy()
print('switch rate per decision', dec.groupby('round_number').switched.mean().round(3).to_dict())
# block sums
d['block'] = d.round_number//4
bs = d.groupby(['session','episode','participant_code','block']).agg(pun=('punishment','sum'), con=('contribution','mean'), gp=('gp','mean'), gc=('gc','mean'), n=('n','mean')).reset_index()
dec = dec.merge(bs, left_on=['session','episode','participant_code', dec.round_number//4], right_on=['session','episode','participant_code','block'], suffixes=('','_b'))
dec['punbin'] = pd.cut(dec.pun, [-1,0,4,10,20,40,500])
print(dec.groupby('punbin').switched.agg(['mean','size']))
dec['gpbin'] = pd.cut(dec.gp_b, [-1,0,1,2.5,5,100])
print('by group mean punishment per round in block'); print(dec.groupby('gpbin').switched.agg(['mean','size']))
dec['own'] = pd.cut(dec.con, [-1,5,10,15,20])
print(dec.groupby(['own','punbin']).switched.mean().unstack().round(2))
# contribution response
x = d[d.round_number < 23].copy()
x['dc'] = x.next_c - x.contribution
x['pb'] = pd.cut(x.punishment, [-1,0,2,5,10,30])
x['cb'] = pd.cut(x.contribution, [-1,5,10,15,20])
print('next-round change in contribution by own c and punishment received (staying players)')
st = x[x.switched==0]
print(st.groupby(['cb','pb']).dc.agg(['mean','size']).unstack().round(2))
# human managers: punishment vs contribution
print(d.groupby(pd.cut(d.contribution,[-1,0,5,10,15,19,20])).punishment.agg(['mean','size']).round(2))
print(d.groupby('round_number').agg(c=('contribution','mean'), p=('punishment','mean')).round(2).T.to_string())
