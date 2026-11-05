"""Is there a group 'punishment climate' effect beyond last round's punishment?
Unpunished players' next-round change vs the group's past punishment."""
import numpy as np, pandas as pd
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d['valid'] = d.contribution_valid.astype(int)
d['c'] = np.where(d.valid == 1, d.contribution, 0)
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
g = d.groupby(['run', 'episode', 'participant_code'])
d['next_c'] = g['c'].shift(-1); d['next_valid'] = g['valid'].shift(-1)
d['own_cum_p'] = g['punishment'].transform(lambda s: s.shift(1).fillna(0).cumsum())
gr = d.groupby(['run', 'episode', 'round_number'])
d['grp_p'] = gr['punishment'].transform('mean')
d['grp_c'] = gr['c'].transform('mean')
ge = d.groupby(['run', 'episode', 'participant_code'])
x = d[(d.valid == 1) & (d.next_valid == 1)].copy()
x['dc'] = x.next_c - x.c
h = x[x.run.str.contains('humanlike') & (x.punishment == 0)].copy()
h['grp_p_bin'] = pd.cut(h.grp_p, [-0.1, 0, 1, 2, 4, 30])
h['own_cum_bin'] = pd.cut(h.own_cum_p / (h.round_number + 1), [-0.1, 0, 0.5, 1, 2, 4, 30])
h['cbin'] = pd.cut(h.c, [-1, 4, 9, 14, 19, 20])
print('unpunished this round: dc by c bin x group mean punishment this round')
print(h.pivot_table(index='cbin', columns='grp_p_bin', values='dc', aggfunc='mean', observed=False).round(2))
print('unpunished this round: dc by c bin x own past punishment per round')
print(h.pivot_table(index='cbin', columns='own_cum_bin', values='dc', aggfunc='mean', observed=False).round(2))
print(h.pivot_table(index='cbin', columns='own_cum_bin', values='dc', aggfunc='size', observed=False))
z = x[x.run.str.contains('zero')].copy(); z['cbin'] = pd.cut(z.c, [-1, 4, 9, 14, 19, 20])
# match on group mean c too
for name, s in [('zero', z), ('humanlike unpunished', h)]:
    s = s.copy(); s['gbin'] = pd.cut(s.grp_c, [0, 8, 12, 16, 20])
    print(name); print(s.pivot_table(index='cbin', columns='gbin', values='dc', aggfunc='mean', observed=False).round(2))
