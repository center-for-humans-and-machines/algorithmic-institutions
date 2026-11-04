"""Which round of the block before a switch decision drives switching (reference sims)?
Logit of switching on own punishment in each of the 4 rounds of the block."""
import pandas as pd, statsmodels.formula.api as smf
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('ah')]
d = d.sort_values(['run','episode','participant_code','round_number'])
d['blk'] = d.round_number//4; d['pos'] = d.round_number%4
w = d.pivot_table(index=['run','episode','participant_code','blk'], columns='pos', values=['punishment','contribution','agent_group'])
w.columns = [f'{a[0][:3]}{a[1]}' for a in w.columns]; w = w.reset_index()
nxt = d[d.pos==0][['run','episode','participant_code','blk','agent_group']].copy(); nxt['blk'] -= 1
w = w.merge(nxt, on=['run','episode','participant_code','blk'])
w['sw'] = (w.agent_group != w.age3).astype(int)
w = w[w.age0==w.age3]
gm = d.groupby(['run','episode','blk','agent_group']).contribution.mean().rename('gc').reset_index()
w = w.merge(gm.rename(columns={'agent_group':'age3'}), on=['run','episode','blk','age3'])
m = smf.logit('sw ~ pun0+pun1+pun2+pun3+con0+con3+gc', data=w).fit(disp=0)
print(m.params.round(3))
