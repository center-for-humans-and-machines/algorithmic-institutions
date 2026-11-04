"""Reference sims: switching after decision rounds vs own punishment, share difference; contribution response."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
g = d.groupby(['run', 'episode', 'participant_code'])
d['next_group'] = g.agent_group.shift(-1)
d['next_c'] = g.contribution.shift(-1)
d['next_valid'] = g.contribution_valid.shift(-1)
d['p4'] = g.punishment.transform(lambda x: x.rolling(4, 1).sum())
d['sw'] = (d.next_group != d.agent_group) & d.next_group.notna()
gg = d.groupby(['run', 'episode', 'round_number', 'agent_group'])
d['members'] = gg.participant_code.transform('size')
d['grp_sum_c'] = gg.contribution.transform('sum')
d['others_mean_c'] = (d.grp_sum_c - d.contribution) / (d.members - 1).replace(0, np.nan)
share = gg.common_good.first().reset_index().rename(columns={'common_good': 'o_share'})
share['agent_group'] = 1 - share.agent_group
d = d.merge(share, on=['run', 'episode', 'round_number', 'agent_group'], how='left')
d['o_share'] = d.o_share.fillna(0)
d['sharediff'] = d.common_good - d.o_share
dec = d[d.round_number.isin([3, 7, 11, 15, 19])]
print('switch rate by round', dec.groupby('round_number').sw.mean().round(3).to_dict())
dec = dec.assign(pbin=pd.cut(dec.punishment, [-1, 0, 3, 8, 30]), p4bin=pd.cut(dec.p4, [-1, 0, 3, 8, 20, 200]),
                 sbin=pd.cut(dec.sharediff, [-100, -8, -3, 0, 3, 8, 100]))
ah = dec[dec.run.str.contains('by ah_vs_ah')]
print('ah_vs_ah: by own punishment at decision\n', ah.groupby('pbin').sw.agg(['mean', 'size']).round(3))
print('ah_vs_ah: by 4-round punishment\n', ah.groupby('p4bin').sw.agg(['mean', 'size']).round(3))
# separate p at decision round from earlier rounds: p4 - p
ah = ah.assign(pearly=ah.p4 - ah.punishment)
print('ah_vs_ah: unpunished at decision, by punishment in the 3 rounds before\n',
      ah[ah.punishment == 0].groupby(pd.cut(ah.pearly, [-1, 0, 3, 8, 100])).sw.agg(['mean', 'size']).round(3))
zz = dec[dec.run.str.contains('zero_vs_zero')]
print('zero_vs_zero: by sharediff\n', zz.groupby('sbin').sw.agg(['mean', 'size']).round(3))
print('zero_vs_zero by members', zz.groupby('members').sw.mean().round(3).to_dict())
X = np.column_stack([np.ones(len(ah)), ah.punishment, ah.pearly, ah.sharediff, ah.contribution, ah.members])
print('ah LPM sw ~ 1,p,p_prev3,sharediff,c,members', np.linalg.lstsq(X, ah.sw.astype(float).values, rcond=None)[0].round(4))
# contribution response in sim (ah_vs_ah, players below groupmates, stayed in group)
v = d[d.run.str.contains('by ah_vs_ah') & (d.contribution_valid) & (d.next_valid == True) & (d.others_mean_c > d.contribution)].dropna(subset=['next_c'])
v = v.assign(dc=v.next_c - v.contribution, defi=v.others_mean_c - v.contribution)
X = np.column_stack([np.ones(len(v)), v.contribution, v.defi, v.punishment])
print('sim dc per punishment point (below-mean players):', np.linalg.lstsq(X, v.dc.values, rcond=None)[0].round(3))
v = v.assign(pbin=pd.cut(v.punishment, [-1, 0, 2, 5, 10, 30]), cbin=pd.cut(v.contribution, [-1, 4, 9, 14, 20]))
print(v.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
w = d[d.run.str.contains('by ah_vs_ah') & (d.contribution_valid) & (d.next_valid == True) & (d.contribution >= 15)].dropna(subset=['next_c'])
w = w.assign(dc=w.next_c - w.contribution)
print('high contributors c>=15 dc by p:', w.groupby(pd.cut(w.punishment, [-1, 0, 2, 5, 30])).dc.agg(['mean', 'size']).round(2).to_dict())
