"""Do artificial humans in the zero group join the ah group less when it punished more
(last round / last block, group total)?"""
import pandas as pd
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.isin(['ah group_switching managed by ah_vs_zero', 'ah group_switching managed by zero_vs_ah'])].copy()
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
k = ['run', 'episode', 'participant_code']
d['pc'] = d.punishment * d.contribution_valid
d['next_g'] = d.groupby(k).agent_group.shift(-1)
d['ahg'] = (d.run.str.contains('by zero_vs_ah')).astype(int)  # index of ah group
gp = d.groupby(['run', 'episode', 'round_number', 'agent_group']).agg(ps=('pc', 'sum'), cs=('contribution', 'sum')).reset_index()
gp = gp.sort_values(['run', 'episode', 'agent_group', 'round_number'])
gp['ps4'] = gp.groupby(['run', 'episode', 'agent_group']).ps.transform(lambda x: x.rolling(4, min_periods=1).sum())
gp = gp.rename(columns={'agent_group': 'og'})
sw = d[d.round_number.isin([3, 7, 11, 15, 19]) & (d.agent_group != d.ahg)].copy()
sw['og'] = sw.ahg
sw = sw.merge(gp, on=['run', 'episode', 'round_number', 'og'])
sw['joined'] = (sw.next_g != sw.agent_group).astype(int)
print('join rate from zero group', sw.joined.mean().round(3))
print(sw.groupby(pd.cut(sw.ps, [-1, 0, 3, 8, 15, 200]), observed=False).joined.agg(['mean', 'count']).round(3))
print(sw.groupby(pd.cut(sw.ps4, [-1, 5, 15, 30, 60, 500]), observed=False).joined.agg(['mean', 'count']).round(3))
print(sw.groupby(pd.cut(sw.cs, [-1, 20, 35, 50, 200]), observed=False).joined.agg(['mean', 'count']).round(3))
