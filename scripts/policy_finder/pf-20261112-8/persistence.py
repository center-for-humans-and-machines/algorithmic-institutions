"""AH: does a punished low contributor's rise persist? Spillover of others' punishment."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['mgr'] = [r.split('by ')[1].split('_vs_')[int(a)] for r, a in zip(d.run, d.agent_group)]
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
key = ['run', 'episode', 'participant_code']
g = d.groupby(key)
for k in [1, 2, 3, 4]:
    d[f'c{k}'] = g.contribution.shift(-k)
    d[f'grp{k}'] = g.agent_group.shift(-k)
gg = d.groupby(['run', 'episode', 'round_number', 'agent_group'])
d['gsize'] = gg.contribution.transform('size')
d['oth_p'] = (gg.punishment.transform('sum') - d.punishment) / (d.gsize - 1).replace(0, np.nan)
d['oth_c'] = (gg.contribution.transform('sum') - d.contribution) / (d.gsize - 1).replace(0, np.nan)
a = d[(d.mgr == 'ah') & d.contribution_valid.astype(bool) & (d.round_number <= 18)].copy()
# stay in same group for 4 rounds, no switch between (exclude switch rounds 3,7,11,15 windows)
a = a[a.round_number % 4 == 0]  # rounds 0,4,8,12,16: next 3 rounds in same group guaranteed
a['cb'] = pd.cut(a.contribution, [-1, 0, 3, 6, 10, 15, 20])
a['pb'] = pd.cut(a.punishment, [-1, 0, 2, 5, 10, 30])
for k in [1, 2, 3]:
    a[f'dc{k}'] = a[f'c{k}'] - a.contribution
print('mean change k rounds later, rows at rounds 0,4,8,12,16 (no switch in window)')
for k in [1, 2, 3]:
    print(f'k={k}'); print(a.pivot_table(index='cb', columns='pb', values=f'dc{k}', aggfunc='mean').round(2))
print(a.pivot_table(index='cb', columns='pb', values='dc1', aggfunc='size'))
# regression on all ah rows with next-round same group
b = d[(d.mgr == 'ah') & d.contribution_valid.astype(bool) & (d.grp1 == d.agent_group) & d.c1.notna()].copy()
cb = pd.get_dummies(pd.cut(b.contribution, [-1, 0, 3, 6, 10, 15, 20]), drop_first=False).astype(float).values
low = (b.contribution <= 6).astype(float).values
X = np.column_stack([cb, b.punishment * low, b.punishment * (1 - low), b.oth_p.fillna(0), b.oth_c.fillna(b.contribution)])
y = (b.c1 - b.contribution).values
coef = np.linalg.lstsq(X, y, rcond=None)[0]
print('\nOLS dc1 ~ cbin dummies + own p (c<=6) + own p (c>6) + others mean p + others mean c')
print(coef.round(3))
