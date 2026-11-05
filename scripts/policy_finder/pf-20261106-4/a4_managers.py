import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-4')
from load import load
import numpy as np, pandas as pd
df, gs = load()
# per game & group label: mean punishment per member-round in rounds 0-3, score = mean pool over 24 rounds (empty 0)
full = gs.pivot_table(index=['game','round_number'], columns='group_id', values='cg').fillna(0).stack().rename('pool').reset_index()
score = full.groupby(['game','group_id']).pool.mean()
early = df[df.round_number<4].groupby(['game','group_id']).agg(p0=('punishment','mean'), c0=('contribution','mean'))
allp = df.groupby(['game','group_id']).agg(pall=('punishment','mean'))
size_end = gs[gs.round_number==23].set_index(['game','group_id']).n
t = early.join(allp).join(score).join(size_end.rename('n_end')).fillna({'n_end':0})
t['p0_other'] = t.groupby('game').p0.transform('sum') - t.p0
t['score_other'] = t.groupby('game').pool.transform('sum') - t.pool
t['less'] = t.p0 < t.p0_other
print(t.groupby('less')[['p0','pall','pool','n_end']].mean().round(2))
print('corr', t[['p0','pall','c0','pool','n_end']].corr().round(2))
# within game: lower-punishing manager wins members?
w = t.reset_index(); w = w[w.group_id==0]
w2 = t.reset_index(); 
pairs = w2.pivot(index='game', columns='group_id', values=['p0','pall','pool','n_end'])
dp = pairs['pall'][0]-pairs['pall'][1]; dn = pairs['n_end'][0]-pairs['n_end'][1]; dpool = pairs['pool'][0]-pairs['pool'][1]
print('within-game corr(dpall, dn_end)', np.corrcoef(dp,dn)[0,1].round(2), 'corr(dpall,dpool)', np.corrcoef(dp,dpool)[0,1].round(2))
print('games where lower-punishing group ends larger:', ((dp<0)&(dn>0)|(dp>0)&(dn<0)).sum(), 'of', (dp!=0).sum(), 'ties in size', (dn==0).sum())
# who punished whom: punishment vs deviation from group mean
df['dev'] = df.contribution - df.mc
print(df.groupby(pd.cut(df.dev,[-20,-6,-3,-1,1,3,20])).punishment.agg(['mean', lambda x:(x>0).mean()]).round(2))
# pool per member-round across contributions in large groups
print(gs.groupby('n').apply(lambda x: pd.Series({'cg':x.cg.mean(),'mc':x.mc.mean(),'mp':x.mp.mean(),'k':len(x)})).round(2))
