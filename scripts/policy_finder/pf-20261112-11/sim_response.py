"""In the reference sims (AH players): how next contribution and switching respond to punishment."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
key = ['run', 'episode', 'participant_code']
d['c_next'] = d.groupby(key).contribution.shift(-1)
d['g_next'] = d.groupby(key).agent_group.shift(-1)
d['valid_next'] = d.groupby(key).contribution_valid.shift(-1)
d['dc'] = d.c_next - d.contribution
v = d[(d.contribution_valid == 1) & (d.valid_next == 1) & ((d.round_number % 4) != 3)].copy()
v['pbin'] = pd.cut(v.punishment, [-1, 0, 2, 5, 10, 30])
v['cbin'] = pd.cut(v.contribution, [-1, 0, 5, 10, 15, 20])
print('== next-round change in contribution (non-switch rounds) by own c and punishment')
print(v.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean', observed=True).round(2))
print(v.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='size', observed=True))
# switching: at decisions after rounds 3,7,..,19, punishment received over the 4-round block
d['blk'] = d.round_number // 4
b = d[d.round_number < 20].groupby(key + ['blk']).agg(p=('punishment', 'sum'), c=('contribution', 'mean'),
    g=('agent_group', 'last'), gn=('g_next', 'last'), np_=('punishment', lambda s: (s > 0).sum())).reset_index()
b['sw'] = (b.g != b.gn).astype(float)
# group sizes & pools in block
gs = d[d.round_number < 20].groupby(['run', 'episode', 'blk', 'round_number', 'agent_group']).agg(n=('participant_code', 'size'), gp=('punishment', 'mean'), gc=('contribution', 'mean')).reset_index()
gs = gs.groupby(['run', 'episode', 'blk', 'agent_group']).mean(numeric_only=True).reset_index()
b = b.merge(gs.rename(columns={'agent_group': 'g'}), on=['run', 'episode', 'blk', 'g'])
o = gs.rename(columns={'agent_group': 'g', 'n': 'on', 'gp': 'ogp', 'gc': 'ogc'}).copy(); o['g'] = 1 - o.g
b = b.merge(o.drop(columns='round_number'), on=['run', 'episode', 'blk', 'g'], how='left').fillna({'on': 0, 'ogp': 0, 'ogc': 0})
b['pbin'] = pd.cut(b.p, [-1, 0, 4, 10, 20, 40, 200])
print('== switch rate by block punishment received (sum over 4 rounds)')
print(b.groupby('pbin', observed=True).sw.agg(['mean', 'size']).round(3))
b['gpbin'] = pd.cut(b.gp - b.ogp, [-30, -2, -0.5, 0.5, 2, 30])
print('== switch rate by own group mean punishment minus other group mean punishment (per player-round)')
print(b.groupby('gpbin', observed=True).sw.agg(['mean', 'size']).round(3))
print('== switch rate, own punishment 0, by group punishment difference')
print(b[b.p == 0].groupby('gpbin', observed=True).sw.agg(['mean', 'size']).round(3))
b['cbin'] = pd.cut(b.c, [-1, 3, 7, 11, 15, 20])
print('== switch rate by own mean contribution x own punishment')
print(b.pivot_table(index='cbin', columns='pbin', values='sw', aggfunc='mean', observed=True).round(3))
b['dgc'] = pd.cut(b.gc - b.ogc, [-20, -3, -1, 1, 3, 20])
print('== switch rate by own group contribution minus other group contribution')
print(b.groupby('dgc', observed=True).sw.agg(['mean', 'size']).round(3))
print('== switch rate by own group size')
print(b.groupby('n').sw.agg(['mean', 'size']).round(3))
import statsmodels.formula.api as smf
b['dn'] = b.n - b.on
m = smf.logit('sw ~ p + c + gp + ogp + gc + ogc + n + blk', data=b).fit(disp=0)
print(m.summary().tables[1])
