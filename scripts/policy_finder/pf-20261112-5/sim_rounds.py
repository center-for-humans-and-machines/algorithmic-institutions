"""Reference sims: per-round contribution, punishment, per-player pool and size of each manager's group."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['pool'] = 1.6 * d.contribution - d.punishment
rows = []
for run, s in d.groupby('run'):
    g0, g1 = run.split('by ')[1].split('_vs_')
    for gid, m in ((0, g0), (1, g1)):
        x = s[s.agent_group == gid].groupby('round_number')
        n = x.size() / s.episode.nunique()
        rows.append(pd.DataFrame({'run': run.split('by ')[1], 'mgr': m, 'c': x.contribution.mean(), 'p': x.punishment.mean(), 'pp': x.pool.mean(), 'n': n, 'tot': x.pool.sum() / s.episode.nunique()}))
r = pd.concat(rows).reset_index()
for run in ['ah_vs_zero', 'zero_vs_zero', 'ah_vs_ah']:
    z = r[r.run == run]
    print(run)
    print(z.pivot_table(index='round_number', columns='mgr' if run == 'ah_vs_zero' else None, values=['c', 'p', 'pp', 'n', 'tot'], aggfunc='mean').round(2).iloc[[0, 1, 2, 3, 4, 7, 8, 12, 16, 19, 20, 21, 22, 23]])
