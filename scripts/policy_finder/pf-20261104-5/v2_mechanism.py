"""v3: same as v1_mechanism.py but for the v2 final points.
punishment per member, pool, for v1 final points (our group = group 0) and the zero/ah refs."""
import pandas as pd, numpy as np
F = 'plots/simulation/policy_finder/pf-20261104-5-v2_sweep/final/per_round.parquet'
R = 'plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet'
d = pd.read_parquet(F)
d['mgr'] = np.where(d.group_id == 0, d.run.str.extract(r'by (\S+)_vs')[0], 'ah')
r = pd.read_parquet(R)
r['mgr'] = [x.split('managed by ')[1].split('_vs_')[g] for x, g in zip(r.run, r.group_id)]
r = r[r.run.str.contains('zero_vs_ah|ah_vs_zero')]
r['mgr'] = r.mgr + '_ref'
d = pd.concat([d, r])
d['pool_c'] = 1.6 * d.contribution
g = d.groupby(['mgr', 'run', 'episode', 'round_number', 'group_id']).agg(
    n=('contribution', 'size'), c=('contribution', 'sum'), p=('punishment', 'sum')).reset_index()
g['pool'] = 1.6 * g.c - g.p
g['cm'] = g.c / g.n; g['pm'] = g.p / g.n
mg = ['pf-20261104-5-v2_gp', 'pf-20261104-5-v2_s050', 'pf-20261104-5-v2_s176', 'zero_ref', 'ah_ref']
blk = pd.cut(g.round_number, [-1, 3, 7, 11, 15, 19, 23], labels=['0-3', '4-7', '8-11', '12-15', '16-19', '20-23'])
for col in ['n', 'cm', 'pm', 'pool']:
    t = g[g.mgr.isin(mg)].groupby([g.mgr, blk], observed=True)[col].mean().unstack().round(2)
    print(col); print(t)
