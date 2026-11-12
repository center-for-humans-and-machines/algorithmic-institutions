"""Reference sims: zero vs ah score, group sizes, contributions per group/round."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['grp'] = d['group_id']
for run, r in d.groupby('run'):
    mgr = run.split('by ')[1].split('_vs_')
    # pool per group per round per episode
    g = r.groupby(['episode', 'round_number', 'grp']).agg(pool=('common_good', 'first'), n=('contribution', 'size'),
        c=('contribution', 'mean'), p=('punishment', 'mean')).reset_index()
    full = g.set_index(['episode', 'round_number', 'grp']).unstack('grp')
    pool = full['pool'].fillna(0); n = full['n'].fillna(0)
    ep = pool.groupby('episode').mean()
    diff = (ep[0] - ep[1])
    print(run, 'pool g0 %.2f g1 %.2f diff %.2f se %.2f' % (ep[0].mean(), ep[1].mean(), diff.mean(), diff.std() / np.sqrt(len(diff))))
    byr = full.groupby('round_number').mean()
    print(pd.DataFrame({'n0': byr['n'][0], 'n1': byr['n'][1], 'c0': byr['c'][0], 'c1': byr['c'][1], 'p0': byr['p'][0], 'p1': byr['p'][1],
                        'pool0': pool.groupby('round_number').mean()[0], 'pool1': pool.groupby('round_number').mean()[1]}).round(2).to_string())
