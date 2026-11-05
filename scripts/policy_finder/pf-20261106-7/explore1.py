import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-7')
from load import load
import numpy as np, pandas as pd
df = load()
print(df.shape, df.game.nunique())
print(df.groupby('round_number')[['contribution','punishment']].mean().round(2).T.to_string())
print('pun dist', df.punishment.describe())
print((df.punishment>0).mean())
# group size by round
gs = df.groupby(['game','round_number','group_id']).size().reset_index(name='n')
print(gs.groupby('round_number').n.apply(lambda x: (x.value_counts(normalize=True).sort_index().round(2).to_dict())).to_string())
