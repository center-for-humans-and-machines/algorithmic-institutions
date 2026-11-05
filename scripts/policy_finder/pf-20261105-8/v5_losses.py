"""v5: where do payoff points go under v3/v4 final points? per-round mean c, p and loss vs full cooperation."""
import pandas as pd, numpy as np, sys
for v in sys.argv[1:]:
    f = pd.read_parquet(f'plots/simulation/policy_finder/{v}_sweep/final/per_round.parquet',
                        columns=['run', 'episode', 'round_number', 'contribution', 'punishment', 'contribution_valid', 'payoff'])
    for run, d in f.groupby('run'):
        d = d[d.contribution_valid.astype(bool)]
        g = d.groupby('round_number')
        c = g.contribution.mean(); p = g.punishment.mean()
        sc = d.groupby(['episode', 'round_number']).payoff.mean().groupby('episode').sum().mean()
        print(run.split(' by ')[1], 'score', round(sc, 1), 'contrib loss', round((0.6 * (20 - c)).sum(), 1), 'pun cost', round((2 * p).sum(), 1))
        print('  c', np.round(c.values, 1)); print('  p', np.round(p.values, 2))
