"""Human data: dedup, punishment policy of managers, response of contributions to punishment."""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session','episode','round_number','participant_code']).copy()
d['game'] = d.session.astype(str)+'_'+d.episode.astype(str)
print('games', d.game.nunique(), 'rows', len(d))
d = d.sort_values(['game','player_id','round_number'])
g = d.groupby(['game','player_id'])
d['c_next'] = g.contribution.shift(-1)
d['grp_next'] = g.group_id.shift(-1)
d['valid'] = 1 - d.player_no_input
v = d[d.valid == 1]
print('\nmean contribution / punishment by round block')
print(v.groupby(v.round_number//4)[['contribution','punishment']].mean().round(2).T)
print('\nhuman manager punishment by contribution bin')
v['cbin'] = pd.cut(v.contribution, [-1,0,4,9,14,19,20])
print(v.groupby('cbin').punishment.agg(['mean','count', lambda x:(x>0).mean()]).round(2))
# relative to group mean
v['gmean'] = v.groupby(['game','round_number','group_id']).contribution.transform('mean')
v['dev'] = v.contribution - v.gmean
print('\npunishment by deviation from group mean')
print(v.groupby(pd.cut(v.dev,[-21,-8,-4,-1,1,4,21])).punishment.agg(['mean','count']).round(2))
# response: next-round change in contribution by punishment, same group, within block (not across switch)
w = v[(v.grp_next == v.group_id) & v.c_next.notna()].copy()
w['dc'] = w.c_next - w.contribution
w['pbin'] = pd.cut(w.punishment, [-1,0,2,5,10,30])
print('\nnext-round change in contribution by own contribution bin x punishment bin')
print(w.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
print(w.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='count'))
