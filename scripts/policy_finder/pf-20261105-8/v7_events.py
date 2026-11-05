"""v7: how often fresh drops (prev c >= 15 -> c < 2) and chronic lows (c<2, prev c<2) happen per episode and round, v6 final gp."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-8-v6_sweep/final/per_round.parquet')
d = d[d.run.str.endswith('_gp')].copy()
d['player'] = d.participant_code.str.split('_').str[0].astype(int)
d = d.sort_values(['episode', 'player', 'round_number'])
n = d.episode.nunique()
C = d.contribution.values.reshape(n, 4, 24).astype(float); V = d.contribution_valid.values.reshape(n, 4, 24).astype(bool)
P = d.punishment.values.reshape(n, 4, 24)
for t in range(1, 24):
    fr = (V[:, :, t] & V[:, :, t - 1] & (C[:, :, t] < 2) & (C[:, :, t - 1] >= 15)).sum() / n
    fr10 = (V[:, :, t] & V[:, :, t - 1] & (C[:, :, t] < 2) & (C[:, :, t - 1] >= 10)).sum() / n
    ch = (V[:, :, t] & (C[:, :, t] < 2) & ~((C[:, :, t - 1] >= 5) & V[:, :, t - 1])).sum() / n
    print(t, f'fresh>=15 {fr:.3f}  fresh>=10 {fr10:.3f}  other lows {ch:.3f}  mean p to lows {P[:, :, t][V[:, :, t] & (C[:, :, t] < 2)].mean() if (V[:, :, t] & (C[:, :, t] < 2)).any() else 0:.1f}')
