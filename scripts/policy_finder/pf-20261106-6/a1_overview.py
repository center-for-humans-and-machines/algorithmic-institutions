import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-6')
from load import load
import numpy as np, pandas as pd
df=load()
print(df.shape, df.game.nunique(), df.columns.tolist())
print(df.groupby('round_number')[['contribution','punishment']].mean().round(2).T.to_string())
g=df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'),c=('contribution','mean'),p=('punishment','mean'),cg=('common_good','first')).reset_index()
print('pool per group-round mean', g.cg.mean(), 'size dist', g.n.value_counts().to_dict())
print(g.groupby('round_number')[['n','cg']].mean().round(1).T.to_string())
print('punish distribution', df.punishment.describe(), (df.punishment>0).mean())
