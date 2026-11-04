"""Reference sims: score (pool = 1.6*sum c - sum p) and membership per manager, per block."""
import pandas as pd
from aimanager.simulation.pool_scores import add_pool, against_anchor
pd.set_option('display.width', 200)
d = add_pool(pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet'))
for m, f in against_anchor(d, 'ah').items():
    print(m, 'pool %.2f se %.2f members %.2f' % (f.pool.mean(), f.pool.std() / len(f) ** .5, f.members.mean()))
g = d.groupby(['run', 'episode', 'round_number', 'agent_group']).agg(
    pool=('pool', 'sum'), n=('pool', 'size'), c=('contribution', 'mean'), p=('punishment', 'mean')).reset_index()
g['blk'] = g.round_number // 4
print(g.pivot_table(index=['run', 'agent_group'], columns='blk', values=['n']).round(2))
print(g.pivot_table(index=['run', 'agent_group'], columns='blk', values=['c']).round(2))
print(g.pivot_table(index=['run', 'agent_group'], columns='blk', values=['p']).round(2))
