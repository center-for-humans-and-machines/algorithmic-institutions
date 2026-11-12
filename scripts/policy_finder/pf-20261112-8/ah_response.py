"""Artificial humans in the reference sims: response to punishment (contribution, switching)."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
key = ['run', 'episode', 'participant_code']
g = d.groupby(key)
d['c_next'] = g.contribution.shift(-1)
d['grp_next'] = g.agent_group.shift(-1)
d['switch'] = (d.grp_next != d.agent_group) & d.grp_next.notna()
gg = d.groupby(['run', 'episode', 'round_number', 'agent_group'])
d['gsize'] = gg.contribution.transform('size')
d['g_c'] = gg.contribution.transform('mean')
d['g_p'] = gg.punishment.transform('mean')
d['mgr'] = [r.split('by ')[1].split('_vs_')[int(a)] for r, a in zip(d.run, d.agent_group)]
print(d.groupby('mgr').contribution.describe().round(2))
d['cbin'] = pd.cut(d.contribution, [-1, 0, 5, 10, 15, 19, 20])
print('\nah punishment by cbin'); print(d[d.mgr == 'ah'].groupby('cbin').punishment.agg(['mean', 'size']).round(2))
nx = d[d.c_next.notna() & ~d.switch & d.contribution_valid.astype(bool)]
nx = nx.assign(dc=nx.c_next - nx.contribution, pbin=pd.cut(nx.punishment, [-1, 0, 3, 8, 30]))
print('\nnext-round dc by cbin x pbin, all groups')
print(nx.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
print(nx.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='size'))
print('\ndc by cbin, unpunished, by manager (zero vs ah group, own p==0)')
print(nx[nx.punishment == 0].pivot_table(index='cbin', columns='mgr', values='dc', aggfunc='mean').round(2))
# distribution of contributions
print('\ncontribution distribution (share) by mgr')
print(d.groupby('mgr').cbin.value_counts(normalize=True).unstack().round(3))
# switching
d['p4'] = g.punishment.transform(lambda s: s.rolling(4, min_periods=1).mean())
d['gp4'] = d.groupby(key).g_p.transform(lambda s: s.rolling(4, min_periods=1).mean())
d['c4'] = g.contribution.transform(lambda s: s.rolling(4, min_periods=1).mean())
sw = d[d.round_number.isin([3, 7, 11, 15, 19])].copy()
print('\nswitch rate by mgr'); print(sw.groupby('mgr').switch.mean().round(3))
sw['p4bin'] = pd.cut(sw.p4, [-1, 0, 1, 3, 6, 30])
print(sw[sw.mgr == 'ah'].groupby('p4bin').switch.agg(['mean', 'size']).round(3))
sw['gp4bin'] = pd.cut(sw.gp4, [-1, 0, 0.5, 1.5, 3, 6, 30])
print(sw[sw.mgr == 'ah'].groupby(['gp4bin']).switch.agg(['mean', 'size']).round(3))
print('ah-group switching, own p4==0, by group p4');
print(sw[(sw.mgr == 'ah') & (sw.p4 == 0)].groupby('gp4bin').switch.agg(['mean', 'size']).round(3))
X = np.column_stack([np.ones(len(sw)), sw.p4, sw.gp4, sw.c4, sw.g_c, sw.gsize])
b = np.linalg.lstsq(X, sw.switch.astype(float).values, rcond=None)[0]
print('LPM switch ~ 1 + own p4 + group p4 + own c4 + group c + size:', b.round(4))
