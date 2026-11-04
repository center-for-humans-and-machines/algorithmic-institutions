"""Reference sims: per-group pool, size, contribution, punishment by run and round."""
import pandas as pd
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['pun_pos'] = d.punishment > 0
g = d.groupby(['run', 'episode', 'round_number', 'group_id']).agg(
    n=('contribution', 'size'), csum=('contribution', 'sum'), psum=('punishment', 'sum'),
    cg=('common_good', 'first')).reset_index()
g['pool'] = 1.6 * g.csum - g.psum
# full grid incl. empty groups
idx = pd.MultiIndex.from_product([g.run.unique(), range(1000), range(24), [0, 1]],
                                 names=['run', 'episode', 'round_number', 'group_id'])
f = g.set_index(['run', 'episode', 'round_number', 'group_id']).reindex(idx).fillna(0).reset_index()
print(f.groupby(['run', 'group_id'])[['pool', 'cg', 'n', 'csum', 'psum']].mean())
f['blk'] = f.round_number // 4
print(f.groupby(['run', 'group_id', 'blk'])[['pool', 'n']].mean().unstack('blk').round(2))
print(d.groupby(['run', 'group_id']).agg(pun=('punishment', 'mean'), punpos=('pun_pos', 'mean'), c=('contribution', 'mean')))
