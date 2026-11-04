"""Human data: next-round contribution change vs punishment received, by own contribution level."""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session','episode','round_number','participant_code'])
d['game'] = d.session.astype(str)+'_'+d.episode.astype(str)
print('games', d.game.nunique(), 'rows', len(d))
d = d.sort_values(['game','player_id','round_number'])
g = d.groupby(['game','player_id'])
d['c_next'] = g.contribution.shift(-1)
d['grp_next'] = g.group_id.shift(-1)
d['valid_next'] = 1-g.player_no_input.shift(-1)
d['dc'] = d.c_next - d.contribution
# group mean contribution of others
gm = d.groupby(['game','round_number','group_id']).contribution.transform('mean')
d['gmean'] = gm
d['rel'] = d.contribution - gm
v = d[(d.player_no_input==0)&(d.valid_next==1)&(d.grp_next==d.group_id)&(d.manager_no_input==0)]
v = v[~v.round_number.isin([3,7,11,15,19,23])]
v['pbin'] = pd.cut(v.punishment,[-1,0,3,6,10,30])
v['cbin'] = pd.cut(v.contribution,[-1,4,9,14,19,20])
print('overall mean c', d[d.player_no_input==0].contribution.mean().round(2))
print('punishment rate', (v.punishment>0).mean().round(3), 'mean', v.punishment.mean().round(2))
print(v.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
print(v.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='count'))
# regression dc ~ p + c + gmean
import numpy.linalg as la
X = np.c_[np.ones(len(v)), v.punishment>0, v.punishment, v.contribution, v.gmean]
b = la.lstsq(X, v.dc, rcond=None)[0]; print('dc ~ 1, any_p, p, c, gmean', b.round(3))
