"""Static cost of the drafted rule on the contributions seen under `zero` in the reference sims (no feedback)."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('zero_vs_zero') & d.contribution_valid]
for thr, rate, ts, sf in [(8, .5, 16, .3), (6, .3, 16, 0), (10, 1, 20, .5), (5, .2, 12, 0)]:
    p = np.floor(np.clip(rate * np.clip(thr - d.contribution, 0, None), 0, 30))
    p = p * np.where(d.round_number % 4 == 3, sf, 1) * (d.round_number < ts)
    p = np.floor(p)
    print(thr, rate, ts, sf, 'mean p/member-round %.2f, share punished %.2f' % (p.mean(), (p > 0).mean()))
