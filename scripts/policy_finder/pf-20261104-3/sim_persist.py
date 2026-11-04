"""Sims, ah-managed groups: does the contribution gain after punishment persist (t+1..t+3)?
And who leaves: by own contribution relative to group, punished or not."""
import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number']).reset_index(drop=True)
k = ['run', 'episode', 'participant_code']
d['mgr'] = np.where(d.agent_group == 0, d.run.str.extract(r'by (\w+)_vs')[0], d.run.str.extract(r'_vs_(\w+)$')[0])
d['gmean'] = d.groupby(['run', 'episode', 'round_number', 'agent_group']).contribution.transform('mean')
d['dev'] = d.contribution - d.gmean
for h in (1, 2, 3):
    d[f'c{h}'] = d.groupby(k).contribution.shift(-h)
    d[f'g{h}'] = d.groupby(k).agent_group.shift(-h)
a = d[(d.mgr == 'ah') & d.contribution_valid & (d.g3 == d.agent_group) & (d.round_number % 4 == 0)]
a = a.assign(cbin=pd.cut(a.contribution, [-1, 0, 4, 9, 14, 20]), pbin=pd.cut(a.punishment, [-1, 0, 2, 5, 10, 30]))
print('rounds 4k (no switch for next 3 rounds): mean c at t+1..t+3 by c bin x p bin')
print(a.groupby(['cbin', 'pbin'], observed=True)[['c1', 'c2', 'c3']].mean().round(2).join(a.groupby(['cbin', 'pbin'], observed=True).size().rename('n')))
# leaving by dev and punishment in the block
d['blk'] = d.round_number // 4
d['pblk'] = d.groupby(k + ['blk']).punishment.transform('sum')
d['cblk'] = d.groupby(k + ['blk']).contribution.transform('mean')
d['gblk'] = d.groupby(['run', 'episode', 'blk', 'agent_group']).contribution.transform('mean')
sw = d[d.round_number.isin([3, 7, 11, 15, 19])].copy()
sw['left'] = d.groupby(k).agent_group.shift(-1).loc[sw.index] != sw.agent_group
sw['devb'] = pd.cut(sw.cblk - sw.gblk, [-21, -5, -2, 2, 5, 21])
sw['pb'] = pd.cut(sw.pblk, [-1, 0, 4, 12, 400])
print('leave rate by block deviation from group mean x block punishment (ah) and zero:')
print(sw[sw.mgr == 'ah'].pivot_table(index='devb', columns='pb', values='left', aggfunc='mean', observed=True).round(3))
print(sw[sw.mgr == 'zero'].groupby('devb', observed=True).left.agg(['mean', 'size']).round(3))
print('leave rate by own block contribution (zero mgr):')
print(sw[sw.mgr == 'zero'].groupby(pd.cut(sw.cblk, [-1, 2, 5, 10, 15, 20]), observed=True).left.mean().round(3))
