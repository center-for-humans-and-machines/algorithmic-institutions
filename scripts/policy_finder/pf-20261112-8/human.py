"""Human data: contribution response to punishment, switching drivers."""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
d = d.sort_values(['session', 'episode', 'player_id', 'round_number'])
key = ['session', 'episode', 'player_id']
g = d.groupby(key)
d['c_next'] = g.contribution.shift(-1)
d['grp_next'] = g.group_id.shift(-1)
d['switch'] = (d.grp_next != d.group_id) & d.grp_next.notna()
# group-level stats
gg = d.groupby(['session', 'episode', 'round_number', 'group_id'])
d['gsize'] = gg.contribution.transform('size')
d['g_c'] = gg.contribution.transform('mean')
d['g_p'] = gg.punishment.transform('mean')
d['others_c'] = (d.g_c * d.gsize - d.contribution) / (d.gsize - 1).replace(0, np.nan)
print('overall c', d.contribution.mean().round(2), 'p', d.punishment.mean().round(2), 'share p>0', (d.punishment > 0).mean().round(2))
print('\npunishment by contribution bin (human managers)')
d['cbin'] = pd.cut(d.contribution, [-1, 0, 5, 10, 15, 19, 20])
print(d.groupby('cbin').punishment.agg(['mean', 'size']).round(2))
print('\nnext-round change in c by cbin x punished-or-not (no switch round)')
nx = d[d.c_next.notna() & ~d.switch]
nx = nx.assign(dc=nx.c_next - nx.contribution, pbin=pd.cut(nx.punishment, [-1, 0, 3, 8, 30]))
print(nx.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
print(nx.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='size'))
# regression of dc on punishment controlling c, others_c
import numpy.linalg as la
X = np.column_stack([np.ones(len(nx)), nx.contribution, nx.others_c.fillna(nx.contribution), nx.punishment, nx.g_p])
b = la.lstsq(X, nx.dc.values, rcond=None)[0]
print('\nOLS dc ~ 1 + c + others_c + own p + group mean p:', b.round(3))
# switching
sw = d[d.round_number.isin([3, 7, 11, 15, 19])]
print('\nswitch rate', sw.switch.mean().round(3))
# per player history in last 4 rounds
d['p4'] = g.punishment.transform(lambda s: s.rolling(4, min_periods=1).mean())
d['gp4'] = d.groupby(key).g_p.transform(lambda s: s.rolling(4, min_periods=1).mean())
sw = d[d.round_number.isin([3, 7, 11, 15, 19])].copy()
sw['p4bin'] = pd.cut(sw.p4, [-1, 0, 1, 3, 6, 30])
print(sw.groupby('p4bin').switch.agg(['mean', 'size']).round(3))
sw['gp4bin'] = pd.cut(sw.gp4, [-1, 0.5, 1.5, 3, 6, 30])
print(sw.groupby('gp4bin').switch.agg(['mean', 'size']).round(3))
X = np.column_stack([np.ones(len(sw)), sw.p4, sw.gp4, sw.contribution, sw.g_c, sw.gsize])
b = la.lstsq(X, sw.switch.astype(float).values, rcond=None)[0]
print('LPM switch ~ 1 + own p4 + group p4 + c + group c + size:', b.round(4))
