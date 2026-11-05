import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-5')
from load import load
import numpy as np, pandas as pd
d = load()
print('games', d.game.nunique())
r = d.groupby('round_number').agg(c=('contribution','mean'), p=('punishment','mean'), pfrac=('punishment', lambda x:(x>0).mean()))
print(r.round(2).T.to_string())
# group-level per round
G = d.groupby(['game','round_number','group_id']).agg(n=('player_id','size'), c=('contribution','mean'), p=('punishment','mean'), pool=('common_good','first')).reset_index()
print('size dist', G.n.value_counts().sort_index().to_dict())
print(G.groupby('n')[['c','p','pool']].mean().round(2))
# switches
s = d[d.round_number.isin([3,7,11,15,19])]
print('switch rate by round', s.groupby('round_number').switch.mean().round(3).to_dict())
