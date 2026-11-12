"""Event study in ah-managed sim groups: future contributions (lags 1..4, same block)
of punished vs unpunished players, matched on c_t, c_{t-1}; and switching vs punishment."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
runs = d.run.str.extract(r'by (\w+)_vs_(\w+)')
d['mgr'] = np.where(d.group_id == 0, runs[0], runs[1])
d['omgr'] = np.where(d.group_id == 0, runs[1], runs[0])
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number']).reset_index(drop=True)
key = ['run', 'episode', 'participant_code']
G = d.groupby(key)
d['c_prev'] = G.contribution.shift(1)
for k in range(1, 5):
    d[f'c{k}'] = G.contribution.shift(-k)
    d[f'g{k}'] = G.group_id.shift(-k)
d['gmean'] = d.groupby(['run', 'episode', 'round_number', 'group_id']).contribution.transform('mean')
x = d[(d.mgr == 'ah') & (d.contribution_valid == 1) & d.c_prev.notna()].copy()
x['pp'] = pd.cut(x.punishment, [-1, 0, 2, 5, 10, 30])
x['cb'] = pd.cut(x.contribution, [-1, 0, 3, 6, 9, 12, 15, 20])
for k in [1, 2, 3, 4]:
    y = x[x[f'g{k}'] == x.group_id]
    # residualise future c on c_t, c_prev, gmean in unpunished, by cbin then compare
    import numpy.linalg as la
    X = np.column_stack([np.ones(len(y)), y.contribution, y.c_prev, y.gmean, y.round_number])
    u = (y.punishment == 0).values
    b = la.lstsq(X[u], y[f'c{k}'].values[u], rcond=None)[0]
    y = y.assign(res=y[f'c{k}'] - X @ b)
    print('lag', k); print(y.pivot_table(index='cb', columns='pp', values='res', aggfunc='mean', observed=True).round(2))
# switching at decision rounds 3,7,11,15,19
s = d[d.round_number.isin([3, 7, 11, 15, 19]) & d.g1.notna()].copy()
s['sw'] = (s.g1 != s.group_id).astype(float)
blk = d.copy(); blk['blk'] = blk.round_number // 4
bp = blk.groupby(key + ['blk']).agg(bp=('punishment', 'mean'), bc=('contribution', 'mean')).reset_index()
s['blk'] = s.round_number // 4
s = s.merge(bp, on=key + ['blk'])
s['bpb'] = pd.cut(s.bp, [-1, 0, 1, 2, 4, 8, 30])
s['bcb'] = pd.cut(s.bc, [-1, 4, 8, 12, 16, 20])
for m in ['ah', 'zero']:
    print('switch rate, own mgr', m)
    print(s[s.mgr == m].pivot_table(index='bcb', columns='bpb', values='sw', aggfunc='mean', observed=True).round(3))
print(s.groupby(['mgr', 'omgr']).sw.mean())
# who moves: contribution of switchers by direction
print(s.groupby(['mgr', 'omgr', 'sw']).bc.mean().round(2))
