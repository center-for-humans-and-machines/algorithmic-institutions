"""How the artificial humans' next contribution responds to punishment (reference sims)."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d['pid'] = d.participant_code.str.split('_').str[0].astype(int)
d = d.sort_values(['run', 'episode', 'pid', 'round_number'])
d['c'] = np.where(d.contribution_valid, d.contribution, np.nan)
g = d.groupby(['run', 'episode', 'pid'])
d['c_next'] = g.c.shift(-1)
d['dc'] = d.c_next - d.c
# group mean of others this round
gm = d.groupby(['run', 'episode', 'round_number']).c.transform('mean')
d['gmean'] = gm
d['dev'] = d.c - gm
h = d[(d.run.str.contains('humanlike')) & d.contribution_valid & d.c_next.notna()]
z = d[(d.run.str.contains('zero')) & d.contribution_valid & d.c_next.notna()]
print('humanlike: mean dc by punishment bin, by own contribution bin')
h['pb'] = pd.cut(h.punishment, [-1, 0, 2, 4, 7, 10, 15, 31])
h['cb'] = pd.cut(h.c, [-1, 0, 4, 8, 12, 16, 19, 20])
print(h.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean').round(2))
print(h.pivot_table(index='cb', columns='pb', values='dc', aggfunc='size'))
z['cb'] = pd.cut(z.c, [-1, 0, 4, 8, 12, 16, 19, 20])
print('zero: dc by own c bin'); print(z.groupby('cb').dc.agg(['mean', 'size']).round(2))
print('humanlike punishment policy: mean p by c bin'); print(h.groupby('cb').punishment.agg(['mean']).round(2))
import statsmodels.formula.api as smf
m = smf.ols('dc ~ punishment + c + gmean + round_number', data=h).fit()
print(m.params.round(3)); 
m2 = smf.ols('dc ~ punishment*c + gmean + round_number', data=h).fit(); print(m2.params.round(3))
