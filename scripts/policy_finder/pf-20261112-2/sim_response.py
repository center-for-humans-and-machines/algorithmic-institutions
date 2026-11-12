"""How do artificial-human contributions respond to punishment in the reference sims?
Next-round change in contribution vs this round's punishment, binned by contribution."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
runs = d.run.str.extract(r'by (\w+)_vs_(\w+)')
d['mgr'] = np.where(d.group_id == 0, runs[0], runs[1])
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
key = ['run', 'episode', 'participant_code']
d['c_next'] = d.groupby(key).contribution.shift(-1)
d['g_next'] = d.groupby(key).group_id.shift(-1)
d['valid_next'] = d.groupby(key).contribution_valid.shift(-1)
# group mean contribution this round (others)
gm = d.groupby(['run', 'episode', 'round_number', 'group_id']).contribution.transform('mean')
d['gmean'] = gm
x = d[(d.mgr == 'ah') & d.c_next.notna() & (d.g_next == d.group_id) & (d.valid_next == 1) & (d.contribution_valid == 1)].copy()
x['dc'] = x.c_next - x.contribution
x['cbin'] = pd.cut(x.contribution, [-1, 0, 4, 8, 12, 16, 19, 20])
x['pbin'] = pd.cut(x.punishment, [-1, 0, 2, 4, 6, 10, 30])
print('n per cell'); print(x.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='size'))
print('mean dc'); print(x.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
# zero groups for baseline
z = d[(d.mgr == 'zero') & d.c_next.notna() & (d.g_next == d.group_id) & (d.valid_next == 1) & (d.contribution_valid == 1)].copy()
z['dc'] = z.c_next - z.contribution
z['cbin'] = pd.cut(z.contribution, [-1, 0, 4, 8, 12, 16, 19, 20])
print('zero-managed dc by cbin'); print(z.groupby('cbin').dc.agg(['mean', 'size']).round(2))
# linear regression in ah groups
import numpy.linalg as la
X = np.column_stack([np.ones(len(x)), x.contribution, x.punishment, x.gmean - x.contribution, (x.punishment > 0)])
b = la.lstsq(X, x.dc, rcond=None)[0]
print('dc ~ 1, c, p, gmean-c, p>0:', b.round(3))
# punishment distribution of ah vs contribution
print('ah punishment by contribution'); print(x.groupby('cbin').punishment.agg(['mean', lambda s: (s > 0).mean()]).round(2))
# the persistence: does punishment at t affect c at t+2.. ? multi-step
for lag in [1, 2, 4]:
    d[f'c_l{lag}'] = d.groupby(key).contribution.shift(-lag)
    d[f'g_l{lag}'] = d.groupby(key).group_id.shift(-lag)
y = d[(d.mgr == 'ah') & (d.contribution_valid == 1)]
for lag in [1, 2, 4]:
    yy = y[y[f'g_l{lag}'] == y.group_id].dropna(subset=[f'c_l{lag}'])
    X = np.column_stack([np.ones(len(yy)), yy.contribution, yy.punishment, yy.gmean])
    print('lag', lag, la.lstsq(X, yy[f'c_l{lag}'], rcond=None)[0].round(3))
