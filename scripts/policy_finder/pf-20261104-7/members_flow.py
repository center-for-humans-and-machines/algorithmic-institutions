"""v1+v2 sweeps: what moves my group's size at a switch decision? Group-level OLS of the member change
(n at 4k+4 minus n at 4k+3) on my group's per-member common_good at the decision round, the other
group's, my punishment per member in the decision round and in the three rounds before, and my size."""
import numpy as np
import pandas as pd
pd.set_option('display.width', 250)
B = 'plots/simulation/policy_finder/'
cols = ['run', 'episode', 'round_number', 'punishment', 'contribution', 'agent_group', 'common_good']
frames = []
for v in ['pf-20261104-7_sweep', 'pf-20261104-7-v2_sweep']:
    for part in ['p1of2', 'p2of2']:
        d = pd.read_parquet(f'{B}{v}/per_round_{part}.parquet', columns=cols, filters=[('episode', '<', 100)])
        d['run'] = d.run.astype('category')
        g = d.groupby(['run', 'episode', 'round_number', 'agent_group'], observed=True).agg(
            n=('contribution', 'size'), c=('contribution', 'mean'), p=('punishment', 'mean'), cg=('common_good', 'first')).reset_index()
        w = g.pivot_table(index=['run', 'episode', 'round_number'], columns='agent_group', values=['n', 'c', 'p', 'cg'], observed=True).fillna(0)
        w.columns = [f'{a}{b}' for a, b in w.columns]
        w = w.reset_index()
        frames.append(w); del d
W = pd.concat(frames)
W = W.sort_values(['run', 'episode', 'round_number'])
gk = W.groupby(['run', 'episode'], observed=True)
W['n0_next'] = gk.n0.shift(-1)
W['p0_prev3'] = gk.p0.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
s = W[W.round_number.isin([3, 7, 11, 15, 19])].dropna(subset=['n0_next']).copy()
s['dn'] = s.n0_next - s.n0
s = s[(s.n0 > 0) & (s.n0 < 8)]
X = np.column_stack([np.ones(len(s)), s.cg0, s.cg1, s.p0, s.p0_prev3.fillna(0), s.c0, s.c1, s.n0, s.round_number])
names = ['1', 'cg_mine', 'cg_other', 'p_mine_now', 'p_mine_prev3', 'c_mine', 'c_other', 'n_mine', 'round']
b, *_ = np.linalg.lstsq(X, s.dn.values, rcond=None)
res = s.dn.values - X @ b
se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var())
print(len(s)); print(pd.DataFrame({'b': b, 'se': se}, index=names).round(4))
s['pb'] = pd.cut(s.p0, [-0.1, 0, 1, 2, 4, 8, 30])
print('member change by my mean punishment in the decision round, by round')
print(s.pivot_table(index='pb', columns='round_number', values='dn', aggfunc='mean', observed=True).round(2))
print(s.pivot_table(index='pb', columns='round_number', values='dn', aggfunc='size', observed=True))
# cross table: decision-round punishment x the three rounds before, first decision vs later ones
s['pbp'] = pd.cut(s.p0_prev3.fillna(0), [-0.1, 0, 1, 3, 6, 30])
s['first'] = s.round_number == 3
print('member change: rows prev3 mean punishment, cols decision-round mean punishment')
for f, ss in s.groupby('first'):
    print('first decision' if f else 'later decisions')
    print(ss.pivot_table(index='pbp', columns='pb', values='dn', aggfunc='mean', observed=True).round(2))
    print(ss.pivot_table(index='pbp', columns='pb', values='dn', aggfunc='size', observed=True))
# by my group size at later decisions: does punishment hurt more in big groups?
L = s[~s['first']].copy()
L['heavy'] = pd.cut(L.p0_prev3.fillna(0) * 3 + L.p0, [-0.1, 0, 4, 12, 200])
print('later decisions: member change by size (rows) x punishment over the 4 rounds (cols)')
print(L.pivot_table(index='n0', columns='heavy', values='dn', aggfunc='mean', observed=True).round(2))
