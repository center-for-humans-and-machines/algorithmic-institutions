"""Per-group pool, size, contribution and punishment in the reference sims."""
import pandas as pd
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
print(d.groupby('run').episode.nunique())
print(d[['agent_group','group_id']].drop_duplicates().head())
for run, g in d.groupby('run'):
    # pool per group/round/episode: 1.6*sum c - sum p over group members
    g = g.copy()
    g['pc'] = g.punishment * g.contribution_valid
    agg = g.groupby(['episode', 'round_number', 'agent_group']).agg(
        csum=('contribution', 'sum'), psum=('pc', 'sum'), n=('contribution', 'size'),
        cg=('common_good', 'first')).reset_index()
    agg['pool'] = 1.6 * agg.csum - agg.psum
    full = agg.set_index(['episode', 'round_number', 'agent_group']).unstack(fill_value=0).stack()
    print(run)
    print(full.groupby('agent_group')[['pool', 'cg', 'n', 'csum', 'psum']].mean().round(2))
    r = full.reset_index()
    r['blk'] = r.round_number // 4
    print(r.pivot_table(index='blk', columns='agent_group', values='n').round(2).T)
