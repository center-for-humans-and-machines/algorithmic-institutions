"""Summarise reference sims: zero vs ah per group, per round."""
import pandas as pd
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['mgr'] = d.run.str.extract(r'by (\w+)_vs_(\w+)').apply(lambda r: None, axis=1)
runs = d.run.str.extract(r'by (\w+)_vs_(\w+)')
d['m0'], d['m1'] = runs[0], runs[1]
d['mgr'] = d.m0.where(d.group_id == 0, d.m1)
d['pairing'] = d.m0 + '_vs_' + d.m1
g = d.groupby(['pairing', 'episode', 'round_number', 'group_id']).agg(
    pool=('common_good', 'first'), n=('contribution', 'size'), c=('contribution', 'mean'),
    p=('punishment', 'mean'), mgr=('mgr', 'first')).reset_index()
# fill empty groups
print(g.groupby(['pairing', 'group_id']).agg(pool=('pool', 'mean'), n=('n', 'mean'), c=('c', 'mean'), p=('p', 'mean')))
# per game pool avg over 24 rounds (empty groups 0)
pg = g.groupby(['pairing', 'episode', 'group_id']).pool.sum() / 24
pg = pg.unstack()
print((pg[0] - pg[1]).groupby('pairing').agg(['mean', 'sem']))
# per round trajectory zero_vs_ah
for pr in ['zero_vs_ah', 'ah_vs_ah', 'zero_vs_zero']:
    x = g[g.pairing == pr].groupby(['round_number', 'group_id'])[['n', 'c', 'p', 'pool']].mean().unstack()
    print(pr); print(x.round(2).to_string())
