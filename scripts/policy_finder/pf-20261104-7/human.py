"""Human data: dedup, basic stats, response of contribution to punishment."""
import pandas as pd, numpy as np
pd.set_option('display.width', 200)
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code']).copy()
d['game'] = d.session + '_' + d.episode.astype(str)
print('games', d.game.nunique(), 'rows', len(d))
d = d.sort_values(['game', 'participant_code', 'round_number'])
gp = d.groupby(['game', 'participant_code'])
d['c_next'] = gp.contribution.shift(-1)
d['g_next'] = gp.group_id.shift(-1)
d['valid'] = 1 - d.player_no_input
v = d[(d.valid == 1)]
# group mean contribution this round (others)
grp = d.groupby(['game', 'round_number', 'group_id'])
d['gsum'] = grp.contribution.transform('sum'); d['gn'] = grp.contribution.transform('size')
d['others'] = (d.gsum - d.contribution) / (d.gn - 1).clip(lower=1)
d['gpun'] = grp.punishment.transform('mean')
print('mean c', d.contribution.mean(), 'mean p', d.punishment.mean(), 'frac p>0', (d.punishment > 0).mean())
print('by round block'); print(d.groupby(d.round_number // 4)[['contribution', 'punishment']].mean().round(2))
# manager policy: punishment by contribution bins
d['cb'] = pd.cut(d.contribution, [-1, 0, 4, 9, 14, 19, 20])
print('manager punishment vs own contribution'); print(d[d.valid == 1].groupby('cb').punishment.agg(['mean', 'count', lambda x: (x > 0).mean()]).round(2))
d['dev'] = d.contribution - d.others
d['db'] = pd.cut(d.dev, [-21, -10, -5, -1, 1, 5, 21])
print('punishment vs deviation from others'); print(d[d.valid == 1].groupby('db').punishment.agg(['mean', 'count']).round(2))
# response: change in contribution next round (same group, not a switch round) vs punishment
s = d[(d.valid == 1) & d.c_next.notna() & (d.g_next == d.group_id) & (d.round_number % 4 != 3)].copy()
s['dc'] = s.c_next - s.contribution
s['pb'] = pd.cut(s.punishment, [-1, 0, 2, 4, 7, 12, 30])
print('next-round change by punishment, split by contribution level')
print(s.pivot_table(index='pb', columns='cb', values='dc', aggfunc='mean').round(2))
print(s.pivot_table(index='pb', columns='cb', values='dc', aggfunc='count'))
import statsmodels.formula.api as smf
try:
    m = smf.ols('dc ~ punishment + contribution + others + gpun + round_number', s).fit()
    print(m.params.round(3), m.bse.round(3))
except Exception as e:
    print(e)
d.to_pickle('/tmp/h.pkl')
