import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-3')
from load import load
import numpy as np, pandas as pd
df = load()
print(df.game.nunique(), len(df))
print(df.groupby('round_number')[['contribution','punishment']].mean().round(2).T.to_string())
# group size dist
gs = df.groupby(['game','round_number','group_id']).size()
print('group size dist', gs.value_counts().sort_index().to_dict())
# per group-round pool
g = df.groupby(['game','round_number','group_id']).agg(c=('contribution','sum'),p=('punishment','sum'),n=('contribution','size'),cg=('common_good','first')).reset_index()
g['pool']=1.6*g.c-g.p
print(g[['pool','cg']].describe())
print('corr pool vs cg', np.corrcoef(g.pool,g.cg)[0,1])
# punishment distribution
print('punishment value counts', df.punishment.value_counts().sort_index().head(31).to_dict())
print('frac punished', (df.punishment>0).mean())
