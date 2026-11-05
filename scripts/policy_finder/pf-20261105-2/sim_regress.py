"""Regressions of next contribution on own/others' punishment under humanlike."""
import numpy as np, pandas as pd
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d['valid'] = d.contribution_valid.astype(int)
d['c'] = np.where(d.valid == 1, d.contribution, 0)
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
key = ['run', 'episode', 'participant_code']
g = d.groupby(key)
d['next_c'] = g['c'].shift(-1); d['next_valid'] = g['valid'].shift(-1)
d['prev_c'] = g['c'].shift(1)
gr = d.groupby(['run', 'episode', 'round_number'])
d['sum_c'] = gr['c'].transform('sum'); d['sum_p'] = gr['punishment'].transform('sum')
d['n_valid'] = gr['valid'].transform('sum')
d['oth_c'] = (d.sum_c - d.c) / (d.n_valid - d.valid).clip(lower=1)
d['oth_p'] = (d.sum_p - d.punishment) / 3
x = d[(d.valid == 1) & (d.next_valid == 1) & d.prev_c.notna() & d.run.str.contains('humanlike')].copy()
x['cbin'] = pd.cut(x.c, [-1, 4, 9, 14, 19, 20])
for cb, xb in x.groupby('cbin', observed=True):
    p = xb.punishment
    X = np.column_stack([np.ones(len(xb)), xb.c, xb.prev_c, xb.oth_c,
                         np.minimum(p, 3), np.clip(p - 3, 0, 7), np.clip(p - 10, 0, 20),
                         xb.oth_p])
    b = np.linalg.lstsq(X, xb.next_c, rcond=None)[0]
    print(cb, len(xb), 'c,prev_c,oth_c, p<=3 slope, p3-10 slope, p>10 slope, oth_p:', b[1:].round(3))
# group level: effect of others' punishment on high contributors
# effect of round on response
x['late'] = x.round_number >= 12
print(x[x.c <= 9].pivot_table(index='late', columns=pd.cut(x[x.c <= 9].punishment, [-1, 0, 5, 10, 30]), values='next_c', aggfunc='mean', observed=False).round(2))
# what fraction of players and rounds are low contributors under zero
z = d[(d.valid == 1) & d.run.str.contains('zero')]
print('zero: share c<=9 by round', z.groupby('round_number').c.apply(lambda s: (s <= 9).mean()).round(2).values)
print('zero: c distribution round 0', np.bincount(z[z.round_number == 0].c.astype(int), minlength=21))
print('zero: c distribution round 10', np.bincount(z[z.round_number == 10].c.astype(int), minlength=21))
