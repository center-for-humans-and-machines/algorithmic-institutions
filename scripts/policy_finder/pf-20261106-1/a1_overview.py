import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-1')
from load import load
import pandas as pd, numpy as np
df = load()
print('games', df.game.nunique(), 'rows', len(df))
print(df.groupby('round_number')[['contribution','punishment']].mean().round(2).T.to_string())
print('timeout rate', df.player_no_input.mean(), 'mgr timeout', df.manager_no_input.mean())
# group-round table
gr = df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'), c=('contribution','sum'), p=('punishment','sum'), cg=('common_good','first')).reset_index()
gr['pool']=1.6*gr.c-gr.p
print('pool check corr', np.corrcoef(gr.pool, gr.cg)[0,1])
# per game, full-length pool per group (empty group 0)
tot = gr.groupby(['game','group_id']).pool.sum()/24
print('mean pool per group per round', tot.mean(), 'sd', tot.std())
print('group size dist by round', gr.groupby('round_number').n.apply(lambda s: (s.max(), s.min())).to_string())
# per group-game: mean punishment, mean contribution, final size
gg = gr.groupby(['game','group_id']).agg(p=('p','sum'), c=('c','sum'), pool=('pool','sum')).reset_index()
gg['nrounds']=24
gg['pool']/=24
last = gr[gr.round_number==23].set_index(['game','group_id']).n
gg['final_n'] = [last.get((a,b),0) for a,b in zip(gg.game,gg.group_id)]
print(gg.describe().round(2))
print(gg[['p','c','pool','final_n']].corr().round(2))
