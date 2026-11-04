"""Reference sims: each manager's group pool and size, per pairing."""
import pandas as pd
from aimanager.simulation.pool_scores import add_pool, group_round_scores, episode_members, pairing_of
d = add_pool(pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet'))
for run in d.run.unique():
    sub = d[d.run == run]
    ep = group_round_scores(sub, 'pool', 'sum')
    g0, g1 = pairing_of(run)
    e = ep.groupby('episode').mean()
    print(run.split('by ')[1], f"g0({g0}) pool {e[0].mean():.2f} se {e[0].std()/len(e)**.5:.2f} members {episode_members(sub,0).mean():.2f} | g1({g1}) pool {e[1].mean():.2f} members {episode_members(sub,1).mean():.2f}")
    # by block of 4 rounds
    blk = ep.reset_index(); blk['b'] = blk.round_number // 4
    print('  per block g0:', blk.groupby('b')[0].mean().round(1).tolist())
    print('  per block g1:', blk.groupby('b')[1].mean().round(1).tolist())
