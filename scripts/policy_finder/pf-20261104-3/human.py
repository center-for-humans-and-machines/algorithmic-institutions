"""Human data: manager style vs pool and size; who leaves; contribution response."""
import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
d['game'] = d.session + '_' + d.episode.astype(str)
print('games', d.game.nunique(), 'rows', len(d))
d['valid'] = 1 - d.player_no_input
d['cval'] = d.contribution * d.valid
g = d.groupby(['game', 'group_id', 'round_number']).agg(
    csum=('cval', 'sum'), psum=('punishment', 'sum'), n=('cval', 'size'), cmean=('contribution', 'mean')).reset_index()
g['pool'] = 1.6 * g.csum - g.psum
full = pd.MultiIndex.from_product([d.game.unique(), [0, 1], range(24)], names=['game', 'group_id', 'round_number'])
g = g.set_index(['game', 'group_id', 'round_number']).reindex(full).fillna({'pool': 0, 'n': 0, 'psum': 0}).reset_index()
gg = g.groupby(['game', 'group_id']).agg(pool=('pool', 'mean'), n=('n', 'mean'), psum=('psum', 'mean'), cmean=('cmean', 'mean'))
gg['p_per_member'] = gg.psum / gg.n.clip(lower=0.1)
print('overall per group: ', gg.mean().round(2).to_dict())
gg['pq'] = pd.qcut(gg.p_per_member, 4)
print(gg.groupby('pq', observed=True)[['pool', 'n', 'psum', 'cmean']].mean().round(2))
# within game: group with less punishment vs more
w = gg.reset_index().pivot(index='game', columns='group_id', values=['pool', 'p_per_member', 'n'])
lessp = np.where(w.p_per_member[0] < w.p_per_member[1], 0, 1)
pl = np.where(lessp == 0, w.pool[0], w.pool[1]); ph = np.where(lessp == 0, w.pool[1], w.pool[0])
nl = np.where(lessp == 0, w.n[0], w.n[1])
print('within game: less-punishing group pool %.1f vs more-punishing %.1f; size %.2f; frac less wins %.2f' % (pl.mean(), ph.mean(), nl.mean(), (pl > ph).mean()))
# human managers' punishment by contribution
d['cbin'] = pd.cut(d.contribution, [-1, 0, 4, 9, 14, 19, 20])
print('human manager punishment by contribution bin (valid):')
v = d[d.valid == 1]
print(v.groupby('cbin', observed=True).punishment.agg(['mean', lambda x: (x > 0).mean(), 'size']).round(2))
print('punishment by rel. contribution (c - group mean):')
v = v.assign(dev=v.contribution - v.groupby(['game', 'group_id', 'round_number']).contribution.transform('mean'))
print(v.groupby(pd.cut(v.dev, [-21, -8, -4, -1, 1, 4, 21]), observed=True).punishment.agg(['mean', 'size']).round(2))
