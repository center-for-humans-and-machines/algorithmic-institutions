"""Human data: punishment response, switching drivers, manager behaviour."""
import numpy as np
import pandas as pd
e = pd.read_csv('experiments/2group_8agent_50ep.csv')
e = e.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
e = e.sort_values(['session', 'episode', 'participant_code', 'round_number'])
print('games', e.groupby(['session', 'episode']).ngroups, 'rows', len(e))
k = ['session', 'episode', 'participant_code']
e['gkey'] = e.session + '_' + e.episode.astype(str) + '_' + e.group_id.astype(str)
grp = e.groupby(['session', 'episode', 'round_number', 'group_id'])
e['n'] = grp.contribution.transform('size')
e['gmean_c'] = grp.contribution.transform('mean')
e['others_c'] = (grp.contribution.transform('sum') - e.contribution) / (e.n - 1).clip(lower=1)
e['next_c'] = e.groupby(k).contribution.shift(-1)
e['next_g'] = e.groupby(k).group_id.shift(-1)
e['dc'] = e.next_c - e.contribution
v = e[(e.player_no_input == 0)]

print('\n== manager punishment by contribution ==')
v2 = v[v.manager_no_input == 0]
print(v2.groupby(pd.cut(v2.contribution, [-1, 0, 5, 10, 15, 19, 20])).punishment.agg(['mean', 'count']).round(2))
print('share of punished (p>0):', (v2.punishment > 0).mean().round(3))
print('punishment by round block:', v2.groupby(v2.round_number // 4).punishment.mean().round(2).tolist())

print('\n== contribution change next round (same group, non-switch rounds) by punishment, by contribution bin ==')
s = v[(v.next_g == v.group_id) & v.next_c.notna() & ~v.round_number.isin([3, 7, 11, 15, 19])].copy()
s['pbin'] = pd.cut(s.punishment, [-1, 0, 2, 5, 10, 30])
s['cbin'] = pd.cut(s.contribution, [-1, 5, 10, 15, 20])
print(s.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
print(s.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='count'))
# regression dc ~ c, others, punishment
import numpy.linalg as la
X = np.c_[np.ones(len(s)), s.contribution, s.others_c, s.punishment, s.punishment ** 2]
b = la.lstsq(X, s.dc, rcond=None)[0]
print('dc ~ 1, c, others_c, p, p^2:', b.round(3))

print('\n== switching ==')
sw = v[v.round_number.isin([3, 7, 11, 15, 19]) & v.next_g.notna()].copy()
sw['left'] = (sw.next_g != sw.group_id).astype(int)
# punishment received in the last 4 rounds
e['p4'] = e.groupby(k).punishment.transform(lambda x: x.rolling(4, min_periods=1).mean())
e['pay4'] = e.groupby(k).payoff.transform(lambda x: x.rolling(4, min_periods=1).mean())
sw = sw.join(e[['p4', 'pay4']])
print('leave rate', sw.left.mean().round(3))
print(sw.groupby(pd.cut(sw.p4, [-1, 0, 1, 3, 6, 30])).left.agg(['mean', 'count']).round(3))
print(sw.groupby(pd.cut(sw.contribution, [-1, 5, 10, 15, 20])).left.agg(['mean', 'count']).round(3))
print(sw.groupby('n').left.agg(['mean', 'count']).round(3))
# group-level: compare pay of other group
rg = e.groupby(['session', 'episode', 'round_number', 'group_id']).agg(gpay=('payoff', 'mean'), gp=('punishment', 'mean')).reset_index()
o = rg.copy(); o['group_id'] = 1 - o.group_id; o = o.rename(columns={'gpay': 'opay', 'gp': 'op'})
sw = sw.merge(o, on=['session', 'episode', 'round_number', 'group_id'], how='left')
rg2 = rg.rename(columns={'gpay': 'mypay', 'gp': 'myp'})
sw = sw.merge(rg2, on=['session', 'episode', 'round_number', 'group_id'], how='left')
sw['dpay'] = sw.mypay - sw.opay.fillna(sw.mypay)
print(sw.groupby(pd.cut(sw.dpay, [-30, -5, -2, 0, 2, 5, 30])).left.agg(['mean', 'count']).round(3))
sw['dp'] = sw.myp - sw.op.fillna(sw.myp)
print(sw.groupby(pd.cut(sw.dp, [-30, -2, -0.5, 0.5, 2, 30])).left.agg(['mean', 'count']).round(3))
X = np.c_[np.ones(len(sw)), sw.p4, sw.contribution, sw.dpay.fillna(0), sw.n]
print('left ~ 1,p4,c,dpay,n', la.lstsq(X, sw.left, rcond=None)[0].round(4))
