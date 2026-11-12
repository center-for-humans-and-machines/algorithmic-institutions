"""Do the artificial humans reproduce the human responses? Use ah-managed groups in ref sims."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['agent_group'] = d.agent_group.astype(int)
d = d[d.run.isin(['ah group_switching managed by zero_vs_ah', 'ah group_switching managed by ah_vs_zero'])].copy()
d['g'] = d.run + d.episode.astype(str)
d = d.sort_values(['g', 'participant_code', 'round_number'])
gg = d.groupby(['g', 'participant_code'])
d['c_next'] = gg.contribution.shift(-1); d['grp_next'] = gg.agent_group.shift(-1)
d['switch'] = (d.grp_next != d.agent_group) & d.grp_next.notna()
x = d[d.contribution_valid.astype(bool) & d.c_next.notna() & (d.round_number % 4 != 3)].copy()
x['dc'] = x.c_next - x.contribution
x['pb'] = pd.cut(x.punishment, [-1, 0, 3, 6, 10, 30]); x['cb'] = pd.cut(x.contribution, [-1, 5, 10, 15, 20])
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean', observed=False).round(2))
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='size', observed=False))
d['blk'] = d.round_number // 4
bl = d.groupby(['g', 'participant_code', 'blk']).agg(pun4=('punishment', 'mean')).reset_index()
gp = d.groupby(['g', 'blk', 'agent_group']).agg(gpun=('punishment', 'mean'), gc=('contribution', 'mean'), gn=('participant_code', 'nunique')).reset_index()
dec = d[d.round_number.isin([3, 7, 11, 15, 19])].merge(bl, on=['g', 'participant_code', 'blk']).merge(gp, on=['g', 'blk', 'agent_group'])
oth = gp.rename(columns={'gpun': 'opun', 'gc': 'oc', 'gn': 'on'}); oth['agent_group'] = 1 - oth.agent_group
dec = dec.merge(oth, on=['g', 'blk', 'agent_group'], how='left')
dec['managed_by_zero'] = ((dec.run.str.contains('zero_vs') & (dec.agent_group == 0)) | (dec.run.str.contains('vs_zero') & (dec.agent_group == 1)))
print('switch rate by manager', dec.groupby('managed_by_zero').switch.mean().round(3))
a = dec[~dec.managed_by_zero]
a = a.assign(pb=pd.cut(a.pun4, [-1, 0, 1, 3, 6, 30]))
print(a.groupby('pb', observed=False).switch.agg(['mean', 'size']).round(3))
z = dec[dec.managed_by_zero].assign(cb=pd.cut(dec.c4 if 'c4' in dec else dec.contribution, [-1, 5, 10, 15, 20]))
import statsmodels.formula.api as smf
dec2 = dec.dropna(subset=['opun']).copy(); dec2['sw'] = dec2.switch.astype(int)
dec2['dpun'] = dec2.gpun - dec2.opun; dec2['dc'] = dec2.gc - dec2.oc; dec2['dn'] = dec2.gn - dec2.on
print(smf.logit('sw ~ pun4 + dpun + dc + dn', dec2).fit(disp=0).summary().tables[1])
print('switch rate by round, zero-managed vs ah-managed')
print(dec.groupby(['round_number', 'managed_by_zero']).switch.mean().unstack().round(3))
