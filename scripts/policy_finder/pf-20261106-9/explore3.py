import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
df = pd.read_pickle('/tmp/df.pkl')
# group-level per game: manager style (mean punishment per member-round) vs pool
full = []
for (g, gid), d in df.groupby(['game','group_id']):
    r = d.groupby('round_number').agg(cg=('common_good','first'), n=('contribution','size'), c=('contribution','mean'), p=('punishment','mean'))
    r = r.reindex(range(24)).fillna({'cg':0,'n':0})
    full.append(dict(game=g, gid=gid, pun=d.punishment.mean(), pun_early=d[d.round_number<8].punishment.mean(),
        pool=r.cg.mean(), n=r.n.mean(), c=d.contribution.mean(), c_late=d[d.round_number>=12].contribution.mean(),
        punpos=(d.punishment>0).mean()))
G = pd.DataFrame(full)
o = G.copy(); o['gid'] = 1-o.gid
G = G.merge(o[['game','gid','pun','pool','n']], on=['game','gid'], suffixes=('','_o'))
print(G[['pun','pun_early','punpos','pool','n','c','c_late']].corr().round(2))
G['pd'] = G.pun - G.pun_o
print('pun diff vs pool, n'); print(G.groupby(pd.qcut(G.pd,5))[['pool','n','c']].mean().round(2))
print('own pun quintile'); print(G.groupby(pd.qcut(G.pun,5))[['pool','n','c','c_late']].mean().round(2))
# punishment targeting: among groups, how strongly targeted at low contributors (corr of p with dev)
df['gmean'] = df.groupby(['game','round_number','group_id']).contribution.transform('mean')
tg = df.groupby(['game','group_id']).apply(lambda d: np.corrcoef(d.contribution-d.gmean, d.punishment)[0,1] if d.punishment.std()>0 else 0).rename('targ').reset_index().rename(columns={'group_id':'gid'})
G = G.merge(tg, on=['game','gid'])
print('targeting quintile'); print(G.groupby(pd.qcut(G.targ,4))[['pool','n','c','pun']].mean().round(2))
import statsmodels.api as sm
X = sm.add_constant(G[['pun','pun_o','targ']]); print(sm.OLS(G.pool, X).fit().summary().tables[1])
X = sm.add_constant(G[['pun','pun_o','targ']]); print(sm.OLS(G.c, X).fit().summary().tables[1])
