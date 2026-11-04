"""Human managers: punishment by contribution and round, and group pool by how much the manager punished."""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session','episode','round_number','participant_code'])
d['game'] = d.session.astype(str)+'_'+d.episode.astype(str)
v = d[(d.player_no_input==0)&(d.manager_no_input==0)]
print('mean punishment by c bin'); print(v.groupby(pd.cut(v.contribution,[-1,4,9,14,19,20]),observed=True).punishment.agg(['mean',lambda x:(x>0).mean()]).round(2))
print('by block', v.groupby(v.round_number//4).punishment.mean().round(2).values)
print('decision rounds vs others', v.groupby(v.round_number%4==3).punishment.mean().round(2).values)
# group-episode: manager's mean punishment vs group's pool (sum over rounds/24) and size
d['pool'] = 1.6*d.contribution - d.punishment
ge = d.groupby(['game','group_id']).agg(p=('punishment','mean'), pool=('pool', lambda x: x.sum()/24), mem=('player_id', lambda x: len(x)/24)).reset_index()
ge['pq'] = pd.qcut(ge.p, 4)
print(ge.groupby('pq',observed=True)[['pool','mem']].mean().round(2))
