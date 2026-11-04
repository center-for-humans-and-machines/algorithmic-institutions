"""In the reference sims (ah-managed groups): next-round contribution and switching vs punishment."""
import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
k = ['run', 'episode', 'participant_code']
d['c_next'] = d.groupby(k).contribution.shift(-1)
d['g_next'] = d.groupby(k).agent_group.shift(-1)
d['v_next'] = d.groupby(k).contribution_valid.shift(-1)
d['mgr'] = np.where(d.agent_group == 0, d.run.str.extract(r'by (\w+)_vs')[0], d.run.str.extract(r'_vs_(\w+)$')[0])
d['gmean'] = d.groupby(['run', 'episode', 'round_number', 'agent_group']).contribution.transform('mean')
d['dev'] = d.contribution - d.gmean
d['cbin'] = pd.cut(d.contribution, [-1, 0, 4, 9, 14, 19, 20])
d['pbin'] = pd.cut(d.punishment, [-1, 0, 2, 5, 10, 30])
# exclude rounds before a switch for the contribution response (same group next round)
same = d[(d.g_next == d.agent_group) & d.contribution_valid & (d.v_next == True)]
print("next contribution by (own contribution bin, punishment bin), ah-managed:")
a = same[same.mgr == 'ah']
print(a.pivot_table(index='cbin', columns='pbin', values='c_next', aggfunc='mean', observed=True).round(2))
print(a.pivot_table(index='cbin', columns='pbin', values='c_next', aggfunc='size', observed=True))
print("zero-managed, next contribution by own contribution bin:")
z = same[same.mgr == 'zero']
print(z.groupby('cbin', observed=True).c_next.agg(['mean', 'size']).round(2))
# switching
sw = d[d.round_number.isin([3, 7, 11, 15, 19])].copy()
sw['left'] = sw.g_next != sw.agent_group
# punishment received in the 4-round block
d['blk'] = d.round_number // 4
pb = d.groupby(k + ['blk']).punishment.sum().rename('pblk')
sw['blk'] = sw.round_number // 4
sw = sw.join(pb, on=k + ['blk'])
sw['pblkbin'] = pd.cut(sw.pblk, [-1, 0, 3, 8, 15, 30, 200])
print("leave rate by manager and block punishment received:")
print(sw.pivot_table(index='pblkbin', columns='mgr', values='left', aggfunc=['mean', 'size'], observed=True).round(3))
print("leave rate by manager and pairing (opponent):")
sw['pair'] = sw.run.str.extract(r'by (\w+)$')[0]
print(sw.groupby(['pair', 'agent_group']).left.mean().round(3))
