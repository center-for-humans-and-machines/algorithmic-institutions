"""Ref sims (artificial humans): switch probability vs punishment by round in the block,
and contribution response to punishment."""
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('ah_vs')]  # player rows; ah manages group 0 in ah_vs_*, use all ah-managed rows
d['mgr'] = [r.split('managed by ')[1].split('_vs_')[g] for r, g in zip(d.run, d.group_id)]
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
k = ['run', 'episode', 'participant_code']
d['g_next'] = d.groupby(k).group_id.shift(-1)
d['c_next'] = d.groupby(k).contribution.shift(-1)
a = d[d.mgr == 'ah']
rows = []
for dec in [3, 7, 11, 15, 19]:
    s = a[a.round_number.between(dec - 3, dec)]
    w = s.pivot_table(index=k, columns='round_number', values='punishment')
    w.columns = ['p0', 'p1', 'p2', 'p3']
    cc = s.pivot_table(index=k, columns='round_number', values='contribution').mean(1).rename('c')
    last = s[s.round_number == dec].set_index(k)
    w = w.join(cc).join((last.g_next != last.group_id).rename('sw')).dropna()
    w['dec'] = dec
    rows.append(w)
w = pd.concat(rows)
X = np.c_[np.ones(len(w)), w[['p0', 'p1', 'p2', 'p3']].values, w.c]
print('n', len(w), 'switch rate', w.sw.mean().round(3))
print('LPM sw ~ 1 + p(r-3) p(r-2) p(r-1) p(r) + c:', lstsq(X, w.sw.astype(float).values, rcond=None)[0].round(4))
print(w.groupby(pd.cut(w[['p0','p1','p2','p3']].sum(1), [-1, 0, 3, 8, 20, 200]), observed=True).sw.agg(['mean', 'size']).round(3))
# zero-managed group switch rate
z = d[(d.mgr == 'zero') & d.round_number.isin([3, 7, 11, 15, 19])].dropna(subset=['g_next'])
print('zero group switch rate', (z.g_next != z.group_id).mean().round(3))
aa = a[a.round_number.isin([3, 7, 11, 15, 19])].dropna(subset=['g_next'])
print('ah group switch rate', (aa.g_next != aa.group_id).mean().round(3))
# contribution response in sim (ah groups, no switch)
x = a[(a.g_next == a.group_id) & ~a.round_number.isin([3, 7, 11, 15, 19, 23])].dropna(subset=['c_next'])
x = x.assign(dc=x.c_next - x.contribution)
x = x.assign(pb=pd.cut(x.punishment, [-1, 0, 2, 5, 10, 30]), cb=pd.cut(x.contribution, [-1, 4, 9, 14, 19, 20]))
print(x.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 21)]:
    s = x[(x.contribution >= lo) & (x.contribution < hi)]
    b = lstsq(np.c_[np.ones(len(s)), s.contribution, s.punishment], s.dc.values, rcond=None)[0]
    print(f'c in [{lo},{hi}) n={len(s)} slope {b[2]:.3f}')
