"""Human data: does the lift in contribution after punishment of a low contributor persist?
Compare c at t+1..t+3 for low contributors (c<=6, below group mean) punished vs not at t,
staying in group, within a phase window; also by contribution relative to group mean."""
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session','episode','round_number','participant_code'])
d['game'] = d.session.astype(str)+'_'+d.episode.astype(str)
d = d.sort_values(['game','participant_code','round_number'])
g = d.groupby(['game','participant_code'])
for k in (1,2,3):
    d[f'c{k}'] = g.contribution.shift(-k)
    d[f'g{k}'] = g.group_id.shift(-k)
gm = d.groupby(['game','round_number','group_id']).contribution.transform('sum')
gn = d.groupby(['game','round_number','group_id']).contribution.transform('size')
d['others'] = (gm-d.contribution)/(gn-1).clip(lower=1)
d['rel'] = d.contribution/d.others.clip(lower=1)
v = d[(d.player_no_input==0)&(d.g3==d.group_id)&(d.g1==d.group_id)&(gn>1)]
v['pb'] = pd.cut(v.punishment,[-1,0,3,8,30])
v['rb'] = pd.cut(v.rel,[-0.01,0.25,0.5,0.75,1.0,1.5,100])
for k in (1,2,3):
    v[f'd{k}'] = v[f'c{k}']-v.contribution
print('mean change in contribution after k rounds, by rel (c / others mean) x punishment')
for k in (1,3):
    print('k=',k); print(v.pivot_table(index='rb',columns='pb',values=f'd{k}',aggfunc='mean').round(2))
print(v.pivot_table(index='rb',columns='pb',values='d1',aggfunc='count'))
print('punishment vs round (humans):'); print(d.groupby(d.round_number).punishment.mean().round(2).values)
