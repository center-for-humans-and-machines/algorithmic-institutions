"""Per-group pools, sizes, contributions in the reference sims (ah vs zero)."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['mgr'] = d.apply(lambda r: r.run.split('by ')[1].split('_vs_')[r.agent_group], axis=1) if False else None
for run, g in d.groupby('run'):
    m0, m1 = run.split('by ')[1].split('_vs_')
    # group pool per episode and round
    gp = g.groupby(['episode', 'round_number', 'agent_group']).agg(
        pool=('common_good', 'first'), n=('contribution', 'size'),
        c=('contribution', 'mean'), p=('punishment', 'mean')).reset_index()
    full = gp.set_index(['episode', 'round_number', 'agent_group'])
    # pool averaged over rounds, empty group = 0
    idx = pd.MultiIndex.from_product([gp.episode.unique(), range(24), [0, 1]], names=full.index.names)
    full = full.reindex(idx).fillna({'pool': 0, 'n': 0})
    ep = full.groupby(['episode', 'agent_group']).pool.mean().unstack()
    diff = ep[0] - ep[1]
    print(run, 'n_ep', len(ep), f'pool0 {ep[0].mean():.2f} pool1 {ep[1].mean():.2f} diff {diff.mean():.2f} se {diff.std()/np.sqrt(len(diff)):.2f}')
    for grp in [0, 1]:
        s = full.xs(grp, level='agent_group').groupby('round_number')[['n', 'c', 'p', 'pool']].mean()
        print(f'  group {grp} ({[m0, m1][grp]}): size by round', s.n.round(2).values[[0, 4, 8, 12, 16, 20, 23]],
              'c', s.c.round(1).values[[0, 4, 8, 12, 16, 20, 23]], 'p', s.p.round(1).values[[0, 4, 8, 12, 16, 20, 23]])

print('\n--- the sweep score (group sum of 1.6c - p) ---')
import sys; sys.path.insert(0, 'src')
from aimanager.simulation.pool_scores import add_pool, against_anchor
add_pool(d)
for m, f in against_anchor(d, 'ah').items():
    diff = f.pool - f.other
    print(m, f'own {f.pool.mean():.2f} other {f.other.mean():.2f} lead {diff.mean():.2f} se {diff.std()/np.sqrt(len(f)):.2f} members {f.members.mean():.2f} vs {f.other_members.mean():.2f}')
