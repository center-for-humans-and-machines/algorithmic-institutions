"""v5: per-round trajectories (members, contribution, punishment, pool) of zero vs ah and of the
earlier versions' finals, my group = agent_group 0 (rule side) / zero's side."""
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
B = 'plots/simulation/policy_finder/'
srcs = {'v1': B + 'pf-20261104-7_sweep/final/per_round.parquet',
        'v3': B + 'pf-20261104-7-v3_sweep/final/per_round.parquet',
        'v4': B + 'pf-20261104-7-v4_sweep/final/per_round.parquet'}
rows = []
for k, f in srcs.items():
    d = pd.read_parquet(f)
    d = d[d.run.str.endswith('_gp_vs_ah')]
    d['src'] = k; rows.append(d)
r = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
print(r.run.unique())
for run in r.run.unique():
    if 'zero_vs_ah' in run or 'ah_vs_ah' in run:
        d = r[r.run == run].copy(); d['src'] = run.split('by ')[-1]; rows.append(d)
d = pd.concat(rows)
d['pool'] = 1.6 * d.contribution - d.punishment
g = d.groupby(['src', 'episode', 'round_number', 'agent_group']).agg(
    n=('pool', 'size'), c=('contribution', 'mean'), p=('punishment', 'mean'), pool=('pool', 'sum')).reset_index()
full = pd.MultiIndex.from_product([g.src.unique(), range(g.episode.max() + 1), range(24), [0, 1]], names=['src', 'episode', 'round_number', 'agent_group'])
# fill empties: pool 0, n 0
g2 = g.set_index(['src', 'episode', 'round_number', 'agent_group'])
eps = g.groupby('src').episode.max()
g2 = g2.reindex(full)
g2 = g2[[e <= eps[s] for s, e in zip(g2.index.get_level_values(0), g2.index.get_level_values(1))]]
g2[['n', 'pool']] = g2[['n', 'pool']].fillna(0)
g2 = g2.reset_index()
for side in [0, 1]:
    print('--- agent_group', side)
    x = g2[g2.agent_group == side]
    for v in ['n', 'c', 'p', 'pool']:
        print(v); print(x.pivot_table(index='src', columns='round_number', values=v).round(1).iloc[:, ::1].to_string())
    print('score', x.groupby('src').pool.mean().round(2).to_dict())
