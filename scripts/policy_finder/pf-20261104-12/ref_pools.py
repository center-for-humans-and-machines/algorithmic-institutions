"""Pool and members of each manager against ah in the reference sims, per round."""
import pandas as pd
from aimanager.simulation.pool_scores import add_pool, against_anchor, group_round_scores
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
add_pool(d)
for m, f in against_anchor(d, 'ah').items():
    print(m, 'pool %.2f se %.2f members %.2f' % (f.pool.mean(), f.pool.std()/len(f)**.5, f.members.mean()))
for run in d.run.unique():
    sub = d[d.run == run]
    s = group_round_scores(sub, 'pool', 'sum').groupby('round_number').mean()
    n = sub.groupby(['episode','round_number','agent_group']).size().unstack(fill_value=0).groupby('round_number').mean()
    c = sub.groupby(['round_number','agent_group']).contribution.mean().unstack()
    p = sub.groupby(['round_number','agent_group']).punishment.mean().unstack()
    print(run); print(pd.concat([s.add_prefix('pool'), n.add_prefix('n'), c.add_prefix('c'), p.add_prefix('p')], axis=1).round(2).iloc[::2].to_string())
