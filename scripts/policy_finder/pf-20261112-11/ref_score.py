"""Score (total pool lead) of zero vs ah in reference sims, plus per-round total pools and sizes."""
import pandas as pd, numpy as np
from aimanager.simulation.pool_scores import add_pool, against_anchor, group_round_scores
d = add_pool(pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet'))
s = against_anchor(d, 'ah')
for m, f in s.items():
    l = f.pool - f.other
    print(m, 'lead %.2f +- %.2f own %.1f other %.1f members %.2f other_members %.2f' % (l.mean(), l.std()/np.sqrt(len(l)), f.pool.mean(), f.other.mean(), f.members.mean(), f.other_members.mean()))
# per round total pool & size, zero_vs_ah
for run, zg in [('ah group_switching managed by zero_vs_ah', 0)]:
    x = d[d.run == run]
    pr = group_round_scores(x, 'pool', 'sum').groupby('round_number').mean()
    n = x.groupby(['episode','round_number','agent_group']).size().unstack().fillna(0).groupby('round_number').mean()
    c = x.groupby(['round_number','agent_group']).contribution.mean().unstack()
    print(pd.concat([pr.add_prefix('pool'), n.add_prefix('n'), c.add_prefix('c')], axis=1).round(2).to_string())
