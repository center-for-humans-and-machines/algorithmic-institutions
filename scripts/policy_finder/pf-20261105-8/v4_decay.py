"""v4: decompose the unpunished decay after the horizon: who drops, by how much, after what."""
import pandas as pd, numpy as np
f = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-8-v2_sweep/final/per_round.parquet')
f = f[f.run.str.endswith('s076')]  # v2 point with horizon 8, no tail
f['player'] = f.participant_code.str.split('_').str[0].astype(int)
f = f.sort_values(['episode', 'player', 'round_number'])
n = f.episode.nunique()
C = f.contribution.values.reshape(n, 4, 24).astype(float); V = f.contribution_valid.values.reshape(n, 4, 24).astype(bool)
C[~V] = np.nan
d = C[:, :, 1:] - C[:, :, :-1]          # change from t-1 to t
rs = slice(9, 23)                        # changes into rounds 10..23 (no punishment since 8)
dd = d[:, :, rs]; prev = C[:, :, 9:23]
oth = (np.nansum(C, 1, keepdims=True) - np.nan_to_num(C)) / np.maximum((~np.isnan(C)).sum(1, keepdims=True) - (~np.isnan(C)), 1)
othp = oth[:, :, 9:23]
ok = ~np.isnan(dd)
x = dd[ok]; print('mean change/round', x.mean().round(3), 'n', ok.sum())
bins = [-21, -10, -5, -2, -0.5, 0.5, 2, 5, 10, 21]
h = np.histogram(x, bins)[0]
for i in range(len(bins) - 1):
    m = (x >= bins[i]) & (x < bins[i + 1])
    print(f'change [{bins[i]},{bins[i+1]}): share {m.mean():.3f}  contrib to mean {x[m].sum()/len(x):+.3f}')
# by own previous level and own vs others
pv = prev[ok]; op = othp[ok]
for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 19), (19, 21)]:
    m = (pv >= lo) & (pv < hi)
    print(f'prev c {lo}-{hi}: share {m.mean():.3f} mean change {x[m].mean():+.2f} contrib {x[m].sum()/len(x):+.3f}')
gap = pv - op
for lo, hi in [(-21, -5), (-5, -1), (-1, 1), (1, 5), (5, 21)]:
    m = (gap >= lo) & (gap < hi)
    print(f'own-others {lo}..{hi}: share {m.mean():.3f} mean change {x[m].mean():+.2f} contrib {x[m].sum()/len(x):+.3f}')
# timeouts: change of others after a timeout
