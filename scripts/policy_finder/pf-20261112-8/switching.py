"""AH switching: low contributors' exit vs own punishment; inflow vs other group's punishment."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['mgr'] = [r.split('by ')[1].split('_vs_')[int(a)] for r, a in zip(d.run, d.agent_group)]
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
key = ['run', 'episode', 'participant_code']
g = d.groupby(key)
d['grp_next'] = g.agent_group.shift(-1)
d['switch'] = (d.grp_next != d.agent_group) & d.grp_next.notna()
d['p4'] = g.punishment.transform(lambda s: s.rolling(4, min_periods=1).mean())
d['c4'] = g.contribution.transform(lambda s: s.rolling(4, min_periods=1).mean())
# group-level 4-round means
gr = d.groupby(['run', 'episode', 'round_number', 'agent_group']).agg(gp=('punishment', 'mean'), gc=('contribution', 'mean'), n=('punishment', 'size')).reset_index()
gr = gr.sort_values(['run', 'episode', 'agent_group', 'round_number'])
gr['gp_last'] = gr.gp  # last-round
sw = d[d.round_number.isin([3, 7, 11, 15, 19])].copy()
o = gr.rename(columns={'agent_group': 'other', 'gp': 'o_gp', 'gc': 'o_gc', 'n': 'o_n'})[['run', 'episode', 'round_number', 'other', 'o_gp', 'o_gc', 'o_n']]
sw['other'] = 1 - sw.agent_group
sw = sw.merge(o, on=['run', 'episode', 'round_number', 'other'], how='left')
s = sw[sw.mgr == 'ah'].copy()
s['c4b'] = pd.cut(s.c4, [-1, 2, 5, 10, 20]); s['p4b'] = pd.cut(s.p4, [-1, 0, 1, 2, 4, 30])
print('ah group: switch rate by own 4-round mean c x own 4-round mean p')
print(s.pivot_table(index='c4b', columns='p4b', values='switch', aggfunc='mean').round(3))
print(s.pivot_table(index='c4b', columns='p4b', values='switch', aggfunc='size'))
z = sw[(sw.mgr == 'zero')].copy()
z['o_gpb'] = pd.cut(z.o_gp, [-1, 0, 1, 2, 4, 30])
print('\nzero-group players: switch-out rate by the ah group last-round mean punishment')
print(z.groupby('o_gpb').switch.agg(['mean', 'size']).round(3))
print('\nswitch rate by round and mgr'); print(sw.pivot_table(index='round_number', columns='mgr', values='switch', aggfunc='mean').round(3))
