"""v4: in never-punish games (gp of v3's final), how often a member of my group gives at least
G below the mean of the other valid members, by G, my group size and block; expected spend of a
flat fine per game."""
import pandas as pd, numpy as np
df = pd.read_parquet('plots/simulation/policy_finder/pf-20261112-2-v3_sweep/final/per_round.parquet')
df = df[df.run.str.contains('_gp')].copy()
df['seat'] = np.where(df.run.str.contains('by pf-'), 0, 1)
d = df[df.group_id == df.seat].copy()
d['cv'] = d.contribution * d.contribution_valid
key = ['seat', 'episode', 'round_number']
g = d.groupby(key).agg(S=('cv', 'sum'), V=('contribution_valid', 'sum'), n=('contribution', 'size'))
d = d.join(g, on=key)
d['oth'] = (d.S - d.cv) / (d.V - d.contribution_valid).clip(lower=1)
d['gap'] = np.where((d.V - d.contribution_valid) > 0, d.oth - d.contribution, -99)
d = d[d.contribution_valid == 1]
ngames = 2000
for G in [4, 6, 8, 10, 12, 14]:
    m = d.gap >= G
    r = d[m & (d.round_number <= 19)]
    print(f'G={G:2d}: share of member-rounds {m.mean():.3f}; fines per game (r0-19) {len(r)/ngames:.2f}; with n>=5 {len(r[r.n>=5])/ngames:.2f}; target c mean {r.contribution.mean():.1f}')
print(d.assign(blk=d.round_number // 4, big=d.gap >= 10).groupby('blk').big.mean().round(3))
print(d.assign(big=d.gap >= 10).groupby(d.n.clip(1, 8)).big.mean().round(3))
