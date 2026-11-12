"""Human data: punishment -> next contribution, switching drivers, manager behaviour."""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code']).copy()
d['game'] = d.session.astype(str) + '_' + d.episode.astype(str)
print('games', d.game.nunique(), 'rows', len(d))
d = d.sort_values(['game', 'participant_code', 'round_number'])
g = d.groupby(['game', 'participant_code'])
d['c_next'] = g.contribution.shift(-1)
d['grp_next'] = g.group_id.shift(-1)
d['switch'] = (d.grp_next != d.group_id) & d.grp_next.notna()
d['pun'] = d.punishment.fillna(0)
ok = (d.player_no_input == 0)
# 1. response: delta contribution next round vs punishment, by contribution level
x = d[ok & d.c_next.notna() & (d.round_number % 4 != 3)].copy()
x['dc'] = x.c_next - x.contribution
x['pb'] = pd.cut(x.pun, [-1, 0, 3, 6, 10, 30])
x['cb'] = pd.cut(x.contribution, [-1, 5, 10, 15, 20])
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean').round(2))
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='size'))
# group stats per round
gs = d.groupby(['game', 'round_number', 'group_id']).agg(n=('contribution', 'size'), c=('contribution', 'mean'),
     p=('pun', 'mean'), pool=('common_good', 'first')).reset_index()
print(gs.groupby('round_number')[['n', 'c', 'p']].mean().round(2).T.to_string())
# 2. switching at decision rounds: player's punishment in the last 4 rounds, own vs other group's avg payoff
dec = d[d.round_number.isin([3, 7, 11, 15, 19])].copy()
d['blk'] = d.round_number // 4
bl = d.groupby(['game', 'participant_code', 'blk']).agg(pun4=('pun', 'mean'), c4=('contribution', 'mean'), pay4=('payoff', 'mean')).reset_index()
gp = d.groupby(['game', 'blk', 'group_id']).agg(gpay=('payoff', 'mean'), gpun=('pun', 'mean'), gc=('contribution', 'mean'), gn=('participant_code', 'nunique')).reset_index()
dec['blk'] = dec.round_number // 4
dec = dec.merge(bl, on=['game', 'participant_code', 'blk'])
dec = dec.merge(gp, on=['game', 'blk', 'group_id'])
oth = gp.rename(columns={'gpay': 'opay', 'gpun': 'opun', 'gc': 'oc', 'gn': 'on'}); oth['group_id'] = 1 - oth.group_id
dec = dec.merge(oth, on=['game', 'blk', 'group_id'], how='left')
print('switch rate', dec.switch.mean().round(3), 'n', len(dec))
dec['pb'] = pd.cut(dec.pun4, [-1, 0, 1, 3, 6, 30])
print(dec.groupby('pb').switch.agg(['mean', 'size']).round(3))
dec['gpb'] = pd.cut(dec.gpun - dec.opun.fillna(0), [-30, -2, -0.5, 0.5, 2, 30])
print('own minus other group mean punishment'); print(dec.groupby('gpb').switch.agg(['mean', 'size']).round(3))
dec['payd'] = pd.cut(dec.gpay - dec.opay, [-30, -3, -1, 1, 3, 30])
print('own minus other group mean payoff'); print(dec.groupby('payd').switch.agg(['mean', 'size']).round(3))
dec['cd'] = pd.cut(dec.gc - dec.oc, [-30, -3, -1, 1, 3, 30])
print('own minus other group mean contribution'); print(dec.groupby('cd').switch.agg(['mean', 'size']).round(3))
import statsmodels.formula.api as smf
dec2 = dec.dropna(subset=['opay']).copy(); dec2['sw'] = dec2.switch.astype(int)
dec2['dpun'] = dec2.gpun - dec2.opun; dec2['dc'] = dec2.gc - dec2.oc; dec2['dn'] = dec2.gn - dec2.on
m = smf.logit('sw ~ pun4 + dpun + dc + c4 + dn', dec2).fit(disp=0); print(m.summary().tables[1])
