"""Reference sims: zero vs ah lead, group sizes, punishment & contributions per group."""
import pandas as pd, numpy as np, sys
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
for run, g in d.groupby('run'):
    names = run.split('by ')[1].split('_vs_')
    # pool per group per round per episode
    pool = g.groupby(['episode', 'round_number', 'group_id']).common_good.first().unstack('group_id').fillna(0)
    size = g.groupby(['episode', 'round_number', 'group_id']).size().unstack('group_id').fillna(0)
    ep = pool.groupby('episode').mean()
    diff = ep[0] - ep[1]
    print(run, 'mean pool g0 %.2f g1 %.2f diff %.2f se %.2f' % (ep[0].mean(), ep[1].mean(), diff.mean(), diff.std() / np.sqrt(len(diff))))
    sz = size.groupby('round_number').mean()
    print('  size g0 by round', sz[0].values[[0, 4, 8, 12, 16, 20, 23]].round(2))
    pg = g.groupby(['group_id', 'round_number'])[['punishment', 'contribution']].mean().unstack('group_id')
    print('  punish g0,g1 r0/12/23', pg['punishment'].loc[[0, 12, 23]].values.round(2).tolist())
    print('  contrib g0,g1 r0/12/23', pg['contribution'].loc[[0, 12, 23]].values.round(2).tolist())
