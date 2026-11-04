"""Reference sims: per-group pool (1.6*sum c - sum p), size, contribution, by manager side."""
import pandas as pd, numpy as np
import sys; sys.path.insert(0,'src')
from aimanager.simulation.pool_scores import add_pool, against_anchor, group_round_scores
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
add_pool(d)
print('agent_group==group_id share', (d.agent_group==d.group_id).mean())
for m, f in against_anchor(d, 'ah').items():
    print(m, 'pool %.2f se %.2f members %.2f' % (f.pool.mean(), f.pool.std()/np.sqrt(len(f)), f.members.mean()))
for run, sub in d.groupby('run'):
    piv = group_round_scores(sub, 'pool', 'sum')
    cnt = sub.groupby(['episode','round_number','agent_group']).size().unstack().reindex(columns=[0,1]).fillna(0)
    print(run)
    print('  pool g0 by round', piv[0].groupby('round_number').mean().round(1).values)
    print('  size g0 by round', cnt[0].groupby('round_number').mean().round(2).values)
    print('  mean c g0 by round', sub[sub.agent_group==0].groupby('round_number').contribution.mean().round(1).values)
