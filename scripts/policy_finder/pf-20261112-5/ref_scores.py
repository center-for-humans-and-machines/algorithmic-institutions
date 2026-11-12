"""Score (total pool lead) of the reference pairings, with aimanager's pool_scores."""
import pandas as pd, numpy as np
from aimanager.simulation.pool_scores import against_anchor, add_pool
d = add_pool(pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet'))
s = against_anchor(d, 'ah')
for m, f in s.items():
    lead = f.pool - f.other
    print(m, 'pool %.2f other %.2f lead %.2f se %.2f members %.2f other_members %.2f' % (
        f.pool.mean(), f.other.mean(), lead.mean(), lead.std() / np.sqrt(len(f)), f.members.mean(), f.other_members.mean()))
# per-capita in zero vs ah decomposition
for run, g in d.groupby('run'):
    x = g.groupby(['agent_group', 'round_number']).agg(c=('contribution', 'mean'), p=('punishment', 'mean'), pool=('pool', 'mean'))
    print(run); print(x.groupby('agent_group').mean().round(2))
