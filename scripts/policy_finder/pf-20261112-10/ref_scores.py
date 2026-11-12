"""Reference sims scored as the sweep scores: total pool 1.6*sum(c)-sum(p) per agent_group."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['pool'] = 1.6 * d.contribution - d.punishment
print('group_id==agent_group share', (d.group_id == d.agent_group).mean())
for run, r in d.groupby('run'):
    g = r.groupby(['episode', 'round_number', 'agent_group']).agg(pool=('pool', 'sum'), n=('pool', 'size'),
         c=('contribution', 'mean'), p=('punishment', 'mean'), v=('contribution_valid', 'mean'))
    full = g.unstack('agent_group')
    pool = full['pool'].reindex(columns=[0, 1]).fillna(0); n = full['n'].reindex(columns=[0, 1]).fillna(0)
    ep = pool.groupby('episode').mean(); diff = ep[0] - ep[1]
    print(f'\n{run}: pool g0 {ep[0].mean():.2f} g1 {ep[1].mean():.2f} diff {diff.mean():.2f} se {diff.std()/np.sqrt(len(diff)):.2f}; members {n[0].mean():.2f} {n[1].mean():.2f}')
    byr = pd.DataFrame({'n0': n.groupby('round_number').mean()[0], 'n1': n.groupby('round_number').mean()[1],
        'c0': full['c'][0].groupby('round_number').mean(), 'c1': full['c'][1].groupby('round_number').mean(),
        'p0': full['p'][0].groupby('round_number').mean(), 'p1': full['p'][1].groupby('round_number').mean(),
        'pool0': pool.groupby('round_number').mean()[0], 'pool1': pool.groupby('round_number').mean()[1]})
    print(byr.iloc[[0, 3, 4, 8, 12, 16, 20, 23]].round(2).to_string())
