"""Reference sims with the scoring metric: group total pool sum(1.6c - p)."""
import pandas as pd, numpy as np
from aimanager.simulation.pool_scores import add_pool, against_anchor, group_round_scores
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['agent_group'] = d['agent_group'].astype(int)
add_pool(d)
s = against_anchor(d, 'ah')
for m, f in s.items():
    diff = f.pool - f.other
    print(m, 'own %.2f other %.2f lead %.2f se %.2f members %.2f vs %.2f' % (
        f.pool.mean(), f.other.mean(), diff.mean(), diff.std()/np.sqrt(len(f)), f.members.mean(), f.other_members.mean()))
# by round, zero vs ah (zero seat), pool and size
for run in ['ah group_switching managed by zero_vs_ah', 'ah group_switching managed by ah_vs_zero']:
    sub = d[d.run == run]
    piv = group_round_scores(sub, 'pool', 'sum').groupby('round_number').mean()
    n = sub.groupby(['episode','round_number','agent_group']).size().unstack().fillna(0).groupby('round_number').mean()
    print(run); print(pd.concat([piv, n], axis=1, keys=['pool','n']).round(2).iloc[::2].to_string())
