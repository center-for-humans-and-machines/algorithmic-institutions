import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from load import load
import pandas as pd, numpy as np
df=load()
print(df.shape, df.game.nunique()); print(df.columns.tolist())
print(df.groupby('round_number')[['contribution','punishment']].mean().round(2).T.to_string())
g=df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'),c=('contribution','mean'),p=('punishment','mean'),pool=('common_good','first')).reset_index()
print('group size dist', g.n.value_counts().sort_index().to_dict())
print('pool per round by size'); print(g.groupby('n')[['c','p','pool']].mean().round(2))
# per-game per-group mean pool vs mean punishment
gg=g.groupby(['game','group_id']).agg(pool=('pool','mean'),p=('p','mean'),c=('c','mean'),n=('n','mean')).reset_index()
gg['pq']=pd.qcut(gg.p,5,duplicates='drop')
print(gg.groupby('pq')[['pool','c','n']].mean().round(2))
print('punishment dist', df.punishment.value_counts().sort_index().head(32).to_dict())
