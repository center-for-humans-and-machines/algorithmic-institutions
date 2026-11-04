"""v2: per-round trajectories of v1's final points (our group vs ah) and the zero_vs_ah reference."""
import sys
import pandas as pd
d = pd.read_parquet('plots/simulation/policy_finder/pf-20261104-1_sweep/final/per_round.parquet')
ref = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
ref = ref[ref.run.str.contains('zero_vs_ah')]
d = pd.concat([d, ref])
for run in d.run.unique():
    sub = d[d.run == run].copy()
    g0, g1 = run.split('managed by ')[1].split('_vs_')
    ne = sub.episode.nunique()
    sub['pool'] = 1.6*sub.contribution - sub.punishment
    rows = []
    for r, x in sub.groupby('round_number'):
        row = {'r': r}
        for g, nm in ((0, 'us'), (1, 'ah')):
            y = x[x.agent_group == g]
            row[nm+'_n'] = len(y)/ne
            row[nm+'_c'] = y.contribution.mean()
            row[nm+'_p'] = y.punishment.mean()
            row[nm+'_pool'] = y.pool.sum()/ne
        rows.append(row)
    t = pd.DataFrame(rows).set_index('r')
    print(run.split('by ')[1], 'pool us %.1f ah %.1f' % (t.us_pool.mean(), t.ah_pool.mean()))
    print(t.round(2).to_string())
