"""Human data: next-round contribution change after punishment; switching drivers;
what human managers did."""
import numpy as np
import pandas as pd
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
print('games', d.groupby(['session', 'episode']).ngroups, 'rows', len(d))
d = d.sort_values(['session', 'episode', 'participant_code', 'round_number'])
k = ['session', 'episode', 'participant_code']
d['c_next'] = d.groupby(k).contribution.shift(-1)
d['g_next'] = d.groupby(k).group_id.shift(-1)
d['dc'] = d.c_next - d.contribution
# group mean contribution this round
gm = d.groupby(['session', 'episode', 'round_number', 'group_id']).contribution.transform('mean')
d['gmean'] = gm
d['dev'] = d.contribution - gm
ok = (d.player_no_input == 0) & d.c_next.notna() & (d.g_next == d.group_id)
x = d[ok & ~d.round_number.isin([3, 7, 11, 15, 19])]
x = x.assign(pb=pd.cut(x.punishment, [-1, 0, 2, 5, 10, 30]),
             cb=pd.cut(x.contribution, [-1, 4, 9, 14, 19, 20]))
print('\nmean dc by punishment bin x contribution bin (same group, no switch)')
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean').round(2))
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='size'))
# regression dc ~ contribution + punishment + dev
import numpy.linalg as la
X = np.c_[np.ones(len(x)), x.contribution, x.punishment, x.dev]
b = la.lstsq(X, x.dc.values, rcond=None)[0]
print('dc = %.2f + %.3f c + %.3f p + %.3f dev' % tuple(b))
# punishment effect for low contributors
for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 21)]:
    s = x[(x.contribution >= lo) & (x.contribution < hi)]
    X = np.c_[np.ones(len(s)), s.contribution, s.punishment]
    b = la.lstsq(X, s.dc.values, rcond=None)[0]
    print(f'c in [{lo},{hi}) n={len(s)} dc slope on p {b[2]:.3f}')
# human managers
print('\nhuman manager punishment by contribution bin')
h = d[d.manager_no_input == 0]
print(h.groupby(pd.cut(h.contribution, [-1, 4, 9, 14, 19, 20])).punishment.agg(['mean', 'size']).round(2))
print('punishment by round', h.groupby('round_number').punishment.mean().round(2).values)
print('contrib by round', d.groupby('round_number').contribution.mean().round(2).values)
d.to_pickle('/tmp/pf5_human.pkl')
