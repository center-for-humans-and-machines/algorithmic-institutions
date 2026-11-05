"""Pilots and humanlike sims: does early punishment (rounds 0-3) go with later contributions?"""
import pandas as pd, numpy as np
df = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
e = df.groupby('episode_id').apply(lambda g: pd.Series({
    'p_early': g[g.round_number < 4].punishment.mean(), 'c0': g[g.round_number == 0].contribution.mean(),
    'c_late': g[g.round_number.between(4, 7)].contribution.mean(), 'p_late': g[g.round_number.between(4, 7)].punishment.mean()}), include_groups=False).dropna()
X = np.column_stack([np.ones(len(e)), e.p_early, e.c0, e.p_late])
print('pilot episodes', len(e), 'c_late(4-7) ~ 1, p_early, c0, p_late', np.linalg.lstsq(X, e.c_late, rcond=None)[0].round(3))
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
h = r[r.run.str.contains('humanlike')]
e = h.groupby('episode').apply(lambda g: pd.Series({
    'p_early': g[g.round_number < 4].punishment.mean(), 'c0': g[g.round_number == 0].contribution.mean(),
    'p_mid': g[g.round_number.between(4, 15)].punishment.mean(),
    'c_late': g[g.round_number >= 16].contribution.mean()}), include_groups=False).dropna()
X = np.column_stack([np.ones(len(e)), e.p_early, e.c0, e.p_mid])
print('humanlike c_late(16-23) ~ 1, p_early, c0, p_mid', np.linalg.lstsq(X, e.c_late, rcond=None)[0].round(3))
