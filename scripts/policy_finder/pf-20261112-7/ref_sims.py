"""Reference sims: zero vs ah. Group pools, sizes, contributions by round."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['agent_group'] = d['agent_group'].astype(int)
for run, r in d.groupby('run'):
    # per episode, round, group: pool and size
    g = r.groupby(['episode', 'round_number', 'agent_group']).agg(
        pool=('common_good', 'first'), n=('contribution', 'size'),
        c=('contribution', 'mean'), p=('punishment', 'mean')).reset_index()
    full = g.set_index(['episode', 'round_number', 'agent_group']).unstack('agent_group')
    pool = full['pool'].fillna(0); n = full['n'].fillna(0)
    lead = (pool[0] - pool[1]).groupby('episode').mean()
    print(run, 'lead g0-g1 %.2f se %.2f' % (lead.mean(), lead.std() / np.sqrt(len(lead))))
    byr = g.groupby(['round_number', 'agent_group'])[['pool', 'n', 'c', 'p']].mean().unstack()
    print(byr.iloc[[0, 3, 4, 8, 12, 16, 20, 23]].round(2).to_string())
