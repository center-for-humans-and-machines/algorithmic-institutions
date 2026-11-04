"""Human data: who switches at the switch decisions, and why."""
import pandas as pd, numpy as np
pd.set_option('display.width', 200)
d = pd.read_pickle('/tmp/h.pkl')
s = d[(d.round_number % 4 == 3) & (d.round_number < 23)].copy()
s['sw'] = (s.g_next != s.group_id).astype(float)
print('switch rate', s.sw.mean(), 'by round', s.groupby('round_number').sw.mean().round(3).to_dict())
# punishment received over the last 4 rounds, own group mean c and punishment, other group's
gp = d.groupby(['game', 'participant_code'])
d['p4'] = gp.punishment.transform(lambda x: x.rolling(4, 1).sum())
d['c4'] = gp.contribution.transform(lambda x: x.rolling(4, 1).mean())
gr = d.groupby(['game', 'round_number', 'group_id']).agg(gc=('contribution', 'mean'), gpm=('punishment', 'mean'), gn=('contribution', 'size')).reset_index()
s = s.merge(d[['game', 'participant_code', 'round_number', 'p4', 'c4']], on=['game', 'participant_code', 'round_number'])
s = s.drop(columns=['gn']).merge(gr, on=['game', 'round_number', 'group_id'])
o = gr.rename(columns={'gc': 'oc', 'gpm': 'opm', 'gn': 'on'}); o['group_id'] = 1 - o.group_id
s = s.merge(o, on=['game', 'round_number', 'group_id'], how='left').fillna({'oc': 0, 'opm': 0, 'on': 0})
s['p4b'] = pd.cut(s.p4, [-1, 0, 3, 8, 20, 200])
print(s.groupby('p4b', observed=True).sw.agg(['mean', 'count']).round(3))
s['c4b'] = pd.cut(s.c4, [-1, 3, 8, 13, 17, 20])
print(s.pivot_table(index='p4b', columns='c4b', values='sw', aggfunc='mean', observed=True).round(2))
import statsmodels.formula.api as smf
m = smf.logit('sw ~ p4 + c4 + gc + oc + gpm + opm + gn', s).fit(disp=0)
print(m.params.round(3)); print(m.bse.round(3))
