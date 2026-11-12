"""Human data: what predicts a player switching groups, and how contributions respond to punishment."""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code']).copy()
d['game'] = d.session + '_' + d.episode.astype(str)
print('games', d.game.nunique(), 'rows', len(d))
d = d.sort_values(['game', 'participant_code', 'round_number'])
g = d.groupby(['game', 'participant_code'])
d['next_group'] = g.group_id.shift(-1)
d['switch'] = (d.next_group != d.group_id) & d.next_group.notna()
# group-level per round
gr = d.groupby(['game', 'round_number', 'group_id']).agg(gp=('punishment', 'mean'), gc=('contribution', 'mean'), n=('contribution', 'size'), gpay=('payoff', 'mean')).reset_index()
# block stats: rounds in the 4-round block before a decision
d['block'] = d.round_number // 4
blk = d.groupby(['game', 'block', 'group_id']).agg(bp=('punishment', 'mean'), bc=('contribution', 'mean'), bpay=('payoff', 'mean')).reset_index()
own = d.groupby(['game', 'participant_code', 'block']).agg(op=('punishment', 'mean'), oc=('contribution', 'mean'), opay=('payoff', 'mean'), grp=('group_id', 'last')).reset_index()
dec = d[d.round_number.isin([3, 7, 11, 15, 19])][['game', 'participant_code', 'block', 'group_id', 'switch', 'round_number']]
dec = dec.merge(own, on=['game', 'participant_code', 'block'])
dec = dec.merge(blk.rename(columns={'group_id': 'gid'}), left_on=['game', 'block', 'group_id'], right_on=['game', 'block', 'gid'])
oth = blk.copy(); oth['gid'] = 1 - oth.group_id
dec = dec.merge(oth[['game', 'block', 'gid', 'bp', 'bc', 'bpay']].rename(columns={'bp': 'xp', 'bc': 'xc', 'bpay': 'xpay'}), on=['game', 'block', 'gid'])
print('switch rate by decision', dec.groupby('round_number').switch.mean().round(3).tolist())
dec['dp'] = dec.bp - dec.xp
dec['dpay'] = dec.bpay - dec.xpay
for col, bins in [('op', [-1, 0, 1, 3, 6, 31]), ('dp', [-31, -2, -0.5, 0.5, 2, 31]), ('dpay', [-30, -4, -1, 1, 4, 30]), ('oc', [-1, 5, 10, 15, 20])]:
    print(col); print(dec.groupby(pd.cut(dec[col], bins)).switch.agg(['mean', 'size']).round(3))
import statsmodels.formula.api as smf
try:
    m = smf.logit('switch ~ op + dp + dpay + oc + C(round_number)', dec.assign(switch=dec.switch.astype(int))).fit(disp=0)
    print(m.params.round(3)); print(m.pvalues.round(3))
except Exception as e: print(e)
# human managers: punishment vs contribution
print('human punishment by contribution bin')
print(d.groupby(pd.cut(d.contribution, [-1, 0, 5, 10, 15, 20])).punishment.agg(['mean', 'size']).round(2))
print('punish by round', d.groupby('round_number').punishment.mean().round(2).tolist())
# contribution change after punishment
d['nc'] = g.contribution.shift(-1)
d['same'] = d.next_group == d.group_id
x = d[d.same & d.nc.notna()]
x = x.assign(dc=x.nc - x.contribution)
print('next-round contribution change by own punishment')
print(x.groupby(pd.cut(x.punishment, [-1, 0, 2, 5, 10, 31])).dc.agg(['mean', 'size']).round(2))
print('...within contribution<10'); y = x[x.contribution < 10]
print(y.groupby(pd.cut(y.punishment, [-1, 0, 2, 5, 10, 31])).dc.agg(['mean', 'size']).round(2))
