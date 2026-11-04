"""Reference sims: per group (group 0 = first-named manager) pool, members, contribution."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['cval'] = d.contribution * d.contribution_valid
g = d.groupby(['run', 'episode', 'round_number', 'agent_group']).agg(
    csum=('cval', 'sum'), psum=('punishment', 'sum'), n=('cval', 'size'),
    cmean=('contribution', 'mean')).reset_index()
g['pool'] = 1.6 * g.csum - g.psum
full = pd.MultiIndex.from_product([d.run.unique(), range(1000), range(24), [0, 1]],
                                  names=['run', 'episode', 'round_number', 'agent_group'])
g = g.set_index(['run', 'episode', 'round_number', 'agent_group']).reindex(full).fillna({'pool': 0, 'n': 0}).reset_index()
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
print(g.groupby(['run', 'agent_group'])[['pool', 'n', 'cmean', 'psum']].mean())
print()
blk = g.assign(blk=g.round_number // 4)
print(blk.groupby(['run', 'agent_group', 'blk'])[['n', 'cmean']].mean().unstack('blk').round(2))
