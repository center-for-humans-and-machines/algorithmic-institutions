"""Human data: punishment -> next contribution, switching, manager behaviour."""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session','episode','round_number','participant_code'])
d['game'] = d.session.astype(str)+'_'+d.episode.astype(str)
print('games', d.game.nunique(), 'rows', len(d))
d = d.sort_values(['game','participant_code','round_number'])
g = d.groupby(['game','participant_code'])
d['c_next'] = g.contribution.shift(-1)
d['grp_next'] = g.group_id.shift(-1)
d['dc'] = d.c_next - d.contribution
d['switch'] = (d.grp_next != d.group_id).astype(float)
v = d[(d.player_no_input==0)]
# per-group-round stats
gr = d.groupby(['game','round_number','group_id']).agg(n=('contribution','size'), c=('contribution','mean'), p=('punishment','mean'), pool=('common_good','first')).reset_index()
gr['phase']=gr.round_number//4
print(gr.groupby('phase')[['n','c','p','pool']].mean().round(2))
# manager punishment vs contribution
print('\nmean punishment by contribution bin')
v['cb']=pd.cut(v.contribution,[-1,0,4,8,12,16,20])
print(v.groupby('cb').punishment.agg(['mean','count',lambda x:(x>0).mean()]).round(2))
# next-round change by punishment, conditioned on contribution bin, same group next round, not switch round
w = v[(v.switch==0)&v.c_next.notna()]
w['pb']=pd.cut(w.punishment,[-1,0,2,5,10,30])
print('\ndelta c next round by contribution bin x punishment bin')
print(w.pivot_table(index='cb',columns='pb',values='dc',aggfunc='mean').round(2))
print(w.pivot_table(index='cb',columns='pb',values='dc',aggfunc='count'))
# switching at decision rounds
s = v[v.round_number.isin([3,7,11,15,19])]
print('\nswitch rate overall', s.switch.mean().round(3))
# punishment received over the last 4 rounds before decision
d['p4'] = g.punishment.transform(lambda x: x.rolling(4,min_periods=1).mean())
d['c4'] = g.contribution.transform(lambda x: x.rolling(4,min_periods=1).mean())
s = d[d.round_number.isin([3,7,11,15,19])&d.grp_next.notna()].copy()
s['p4b']=pd.cut(s.p4,[-1,0,1,3,6,30])
s['c4b']=pd.cut(s.c4,[-1,4,8,12,16,20])
print(s.pivot_table(index='c4b',columns='p4b',values='switch',aggfunc='mean').round(2))
print(s.pivot_table(index='c4b',columns='p4b',values='switch',aggfunc='count'))
