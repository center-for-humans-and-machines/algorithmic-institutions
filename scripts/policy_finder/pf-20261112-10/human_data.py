"""Human data: contribution response to punishment, switching drivers."""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
d['game'] = d.session + '_' + d.episode.astype(str)
print('games', d.game.nunique(), 'rows', len(d))
d = d.sort_values(['game', 'participant_code', 'round_number'])
g = d.groupby(['game', 'participant_code'])
d['c_next'] = g.contribution.shift(-1)
d['grp_next'] = g.group_id.shift(-1)
d['dc'] = d.c_next - d.contribution
# group mean contribution this round (others)
gs = d.groupby(['game', 'round_number', 'group_id'])
d['n'] = gs.contribution.transform('size')
d['gmean'] = gs.contribution.transform('mean')
d['others'] = (d.gmean * d.n - d.contribution) / (d.n - 1).replace(0, np.nan)
ok = (d.player_no_input == 0)
x = d[ok & d.c_next.notna() & ~d.round_number.isin([3, 7, 11, 15, 19])].copy()
x = x[x.grp_next == x.group_id]
x['pbin'] = pd.cut(x.punishment, [-1, 0, 2, 5, 10, 30])
x['cbin'] = pd.cut(x.contribution, [-1, 5, 10, 15, 19, 20])
print('\nmean next-round change in contribution by own contribution bin x punishment bin')
print(x.pivot_table('dc', 'cbin', 'pbin', aggfunc='mean').round(2))
print(x.pivot_table('dc', 'cbin', 'pbin', aggfunc='size'))
# relative contribution
x['rel'] = x.contribution - x.others
x['rbin'] = pd.cut(x.rel, [-21, -5, -1, 1, 5, 21])
print('\ndc by relative contribution bin x punishment bin')
print(x.pivot_table('dc', 'rbin', 'pbin', aggfunc='mean').round(2))
print(x.pivot_table('dc', 'rbin', 'pbin', aggfunc='size'))
# Human managers' punishment by contribution
print('\nhuman punishment by contribution bin, by round block')
d['blk'] = d.round_number // 4
d['cbin'] = pd.cut(d.contribution, [-1, 0, 5, 10, 15, 19, 20])
print(d[ok].pivot_table('punishment', 'cbin', 'blk', aggfunc='mean').round(2))
print('overall contribution by block', d[ok].groupby('blk').contribution.mean().round(2).tolist())
print('overall punishment by block', d[ok].groupby('blk').punishment.mean().round(2).tolist())
