"""Human managers: their punishment policy and what it got them (one copy per game)."""
import pandas as pd, numpy as np
h = pd.read_csv('experiments/2group_8agent_50ep.csv').drop_duplicates(['session','episode','round_number','participant_code'])
h['pool'] = 1.6*h.contribution - h.punishment
h['gmean'] = h.groupby(['session','episode','round_number','group_id']).contribution.transform('mean')
h['gap'] = h.gmean - h.contribution
# policy: P(punish) and mean punishment by contribution and by gap
print(h.groupby(pd.cut(h.contribution,[-1,0,4,9,14,19,20])).punishment.agg(['mean',lambda x:(x>0).mean(),'size']).round(2))
print(h.groupby(pd.cut(h.gap,[-21,-5,-2,0,2,5,10,21])).punishment.agg(['mean',lambda x:(x>0).mean(),'size']).round(2))
print(h.groupby(h.round_number//4).punishment.mean().round(2))
# per game-group: mean punishment vs pool and members
G = h.groupby(['session','episode','group_id']).agg(p=('punishment','mean'), c=('contribution','mean'), rows=('pool','size'), pool=('pool','sum')).reset_index()
G['members'] = G.rows/24; G['pool'] = G.pool/24
G['pq'] = pd.qcut(G.p, 4)
print(G.groupby('pq', observed=True)[['p','c','members','pool']].mean().round(2))
# within game: does the less-punishing group get the bigger pool?
w = G.pivot_table(index=['session','episode'], columns='group_id', values=['p','pool','members'])
dp = w['p'][0]-w['p'][1]; dpool = w['pool'][0]-w['pool'][1]; dm = w['members'][0]-w['members'][1]
print('corr(dp,dpool)', np.corrcoef(dp,dpool)[0,1].round(2), 'corr(dp,dmembers)', np.corrcoef(dp,dm)[0,1].round(2), 'n games', len(dp))
print('less-punishing group has larger pool in', ((dp<0)==(dpool>0)).mean().round(2))
