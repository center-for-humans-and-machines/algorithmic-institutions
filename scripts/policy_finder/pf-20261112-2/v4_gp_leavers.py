"""v4: in never-punish gp games, who leaves my group and who joins at each decision,
by their block-mean contribution; and ah members' fines vs their leaving."""
import pandas as pd, numpy as np
df = pd.read_parquet('plots/simulation/policy_finder/pf-20261112-2-v3_sweep/final/per_round.parquet')
df = df[df.run.str.contains('_gp')].copy()
df['seat'] = np.where(df.run.str.contains('by pf-'), 0, 1)
df['mine'] = (df.group_id == df.seat).astype(int)
df['blk'] = df.round_number // 4
df = df.sort_values(['seat', 'episode', 'participant_code', 'round_number'])
b = df.groupby(['seat', 'episode', 'participant_code', 'blk']).agg(mine=('mine', 'first'), cm=('contribution', 'mean'), pm=('punishment', 'mean'), c3=('contribution', 'last')).reset_index()
b['mine_next'] = b.groupby(['seat', 'episode', 'participant_code']).mine.shift(-1)
b = b[b.blk < 5]
b['moved'] = (b.mine != b.mine_next).astype(int)
b['cb'] = pd.cut(b.cm, [-1, 1, 4, 8, 12, 16, 20])
print('leave rate by block-mean contribution, my members vs ah members')
print(b.pivot_table(index='cb', columns='mine', values='moved', aggfunc='mean', observed=True).round(3))
print(b.pivot_table(index='cb', columns='mine', values='moved', aggfunc='size', observed=True))
print('by block, mine:')
print(b.pivot_table(index='blk', columns='mine', values='moved', aggfunc='mean').round(3))
print('mean contribution of leavers vs stayers by group')
print(b.groupby(['mine', 'moved']).cm.mean().round(2))
a = b[b.mine == 0].copy(); a['pb'] = pd.cut(a.pm, [-1, 0.01, 1, 3, 6, 30])
print('ah members leave rate by their mean fine in block')
print(a.groupby('pb', observed=True).moved.agg(['mean', 'size']).round(3))
