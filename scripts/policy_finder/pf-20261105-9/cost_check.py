"""Punishment the rule would deal on the round-0 contributions of the zero sim (first-round cost)."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
x = d[d.run.str.contains('zero') & (d.round_number == 0) & d.contribution_valid].contribution.values
print('round-0 c quantiles', np.quantile(x, [.1, .25, .5, .75, .9]))
for target, slope, cap in [(15, 1, 5), (18, 0.5, 3), (12, 2, 10), (20, 1, 30)]:
    p = np.floor(np.minimum(slope * np.clip(target - x, 0, None), cap))
    print(target, slope, cap, 'mean p', p.mean().round(2), 'payoff loss/player', (2 * p.mean()).round(2))
