"""How much would the draft rule punish, per member-round, on the zero group's contributions in the
reference sims (zero vs ah, zero's seat), for a few (k, thr)? Sizes the k range."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.endswith('zero_vs_ah') & (d.agent_group == 0) & (d.contribution_valid == 1)].copy()
key = ['episode', 'round_number']
s = d.groupby(key).contribution.transform('sum'); n = d.groupby(key).contribution.transform('size')
d['mo'] = np.where(n > 1, (s - d.contribution) / (n - 1).clip(lower=1), np.nan)
for thr in (0.5, 0.8, 1.0, 1.2):
    sf = (thr * d.mo - d.contribution).clip(lower=0).fillna(0)
    print(f'thr {thr}: share short {(sf > 0).mean():.2f}, mean shortfall when short {sf[sf > 0].mean():.2f}', end='; ')
    for k in (0.1, 0.3, 0.6, 1.0, 2.0):
        p = np.floor((k * sf).clip(upper=30))
        print(f'k{k}: p/member {p.mean():.2f} punished {(p > 0).mean():.2f}', end='  ')
    print()
