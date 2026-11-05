import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-4')
from load import load
import numpy as np, pandas as pd
df, gs = load()
print('games', df.game.nunique(), 'rows', len(df))
print('timeouts', df.player_no_input.mean(), 'mgr timeouts', df.manager_no_input.mean())
print(df.groupby('round_number').agg(c=('contribution','mean'), p=('punishment','mean')).T.round(2).to_string())
print('group sizes by round'); print(gs.groupby('round_number').n.describe()[['mean','std','min','max']].T.round(2).to_string())
gs['pool_score'] = gs.cg
# score per group per game = mean over rounds of pool (empty=0)
full = gs.pivot_table(index=['game','round_number'], columns='group_id', values='cg').fillna(0)
print('mean pool per group-round', full.values.mean())
# distribution of manager punishment
print('punishment dist', df.punishment.value_counts(normalize=True).sort_index().head(15).round(3).to_dict())
print('frac players punished>0', (df.punishment>0).mean())
