"""Reference sims: per group pool, size, contributions by run and round."""
import pandas as pd
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
print(d[['agent_group','group_id']].drop_duplicates().head())
print(d.groupby(['run','agent_group','group_id']).size())
g = d.groupby(['run', 'episode', 'round_number', 'group_id']).agg(
    n=('contribution', 'size'), c=('contribution', 'sum'), p=('punishment', 'sum'),
    cg=('common_good', 'first')).reset_index()
g['pool'] = 1.6 * g.c - g.p
print(g[['pool','cg']].head())
# score: pool averaged over rounds incl empty groups (fill 0)
full = g.set_index(['run','episode','round_number','group_id']).pool.unstack('group_id', fill_value=0).stack().rename('pool').reset_index()
print(full.groupby(['run','group_id']).pool.mean())
sz = g.set_index(['run','episode','round_number','group_id']).n.unstack('group_id', fill_value=0)
print(sz.groupby(['run','round_number']).mean().unstack('run').iloc[::4])
print((g.c/g.n).groupby([g.run,g.group_id]).mean())
print('---- zero vs ah: per round, by manager side')
g['mgr'] = [r.split('managed by ')[1].split('_vs_')[gid] for r, gid in zip(g.run, g.group_id)]
g['opp'] = [r.split('managed by ')[1].split('_vs_')[1-gid] for r, gid in zip(g.run, g.group_id)]
m = g[g.mgr != g.opp]
t = m.groupby(['mgr','round_number']).agg(n=('n','mean'), pool=('pool','mean'), c=('c','sum'), nn=('n','sum'), p=('p','sum'))
t['cpm'] = t.c/t.nn; t['ppm'] = t.p/t.nn
print(t.drop(columns=['c','nn','p']).unstack('mgr').round(2).to_string())
