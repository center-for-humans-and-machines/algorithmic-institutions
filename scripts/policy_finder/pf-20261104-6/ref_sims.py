"""Reference sims: per side, group pool, size, contributions, punishments by round."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
print(d.groupby('run').episode.nunique())
print(d[['agent_group','group_id']].drop_duplicates().head())
for run, g in d.groupby('run'):
    # group-level: one row per episode, round, group
    gg = g.groupby(['episode','round_number','group_id']).agg(pool=('common_good','first'), n=('participant_code','size'),
          c=('contribution','mean'), pun=('punishment','mean')).reset_index()
    # fill empty groups with 0
    full = pd.MultiIndex.from_product([gg.episode.unique(), range(24), [0,1]], names=['episode','round_number','group_id'])
    gg = gg.set_index(['episode','round_number','group_id']).reindex(full).fillna({'pool':0,'n':0}).reset_index()
    print(run)
    print(gg.groupby('group_id')[['pool','n','c','pun']].mean().round(2))
    print(gg.groupby(['group_id', gg.round_number//4])['n'].mean().unstack().round(2))

print("\n=== total pool (1.6*sum c - sum p over valid members), empty group = 0 ===")
d['cv'] = d.contribution * d.contribution_valid
d['pv'] = d.punishment * d.contribution_valid
for run, g in d.groupby('run'):
    gg = g.groupby(['episode','round_number','group_id']).agg(c=('cv','sum'), p=('pv','sum'), n=('participant_code','size')).reset_index()
    gg['pool'] = 1.6*gg.c - gg.p
    tot = gg.groupby(['group_id']).pool.sum() / (g.episode.nunique()*24)
    print(run, tot.round(2).to_dict())
    print('  by round block', gg.groupby(['group_id', gg.round_number//4]).pool.sum().unstack().div(g.episode.nunique()*4).round(1).to_dict('index'))
