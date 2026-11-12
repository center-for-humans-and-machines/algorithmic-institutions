"""Reference sims: how AH contributions answer punishment, and switching vs own punishment."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
g = d.groupby(['run', 'episode', 'participant_code'])
for k in (1, 2, 3):
    d[f'nc{k}'] = g.contribution.shift(-k)
d['ng'] = g.agent_group.shift(-1)
d['stay'] = d.ng == d.agent_group
pc = lambda s: s.groupby(pd.cut(d.loc[s.index, 'punishment'], [-1, 0, 2, 5, 10, 31]), observed=True)
ah = d[d.run.str.contains('ah_vs_ah')]
print('pool per player by round: ah_vs_ah vs zero_vs_zero')
for r in ['ah_vs_ah', 'zero_vs_zero']:
    s = d[d.run.str.contains(r)]
    print(r, s.groupby('round_number').contribution.mean().round(2).values[[0, 4, 8, 12, 16, 19, 20, 21, 22, 23]])
x = ah[ah.stay & ah.nc1.notna() & (ah.round_number < 21)].copy()
for k in (1, 2, 3):
    x[f'dc{k}'] = x[f'nc{k}'] - x.contribution
for lo, hi in [(0, 5), (5, 10), (10, 21)]:
    y = x[(x.contribution >= lo) & (x.contribution < hi)]
    print(f'c in [{lo},{hi}) delta c at +1,+2,+3 by own punishment')
    print(y.groupby(pd.cut(y.punishment, [-1, 0, 2, 5, 10, 31]), observed=True)[['dc1', 'dc2', 'dc3']].agg('mean').round(2).assign(n=y.groupby(pd.cut(y.punishment, [-1, 0, 2, 5, 10, 31]), observed=True).size()))
# switching vs own block punishment in sims
d['block'] = d.round_number // 4
dec = d[d.round_number.isin([3, 7, 11, 15, 19])].copy()
bp = d.groupby(['run', 'episode', 'participant_code', 'block']).punishment.mean().rename('bp')
dec = dec.join(bp, on=['run', 'episode', 'participant_code', 'block'])
dec['sw'] = ~dec.stay
print('sim switch rate by own block punishment (all runs)')
print(dec.groupby(pd.cut(dec.bp, [-1, 0, 1, 3, 6, 31]), observed=True).sw.agg(['mean', 'size']).round(3))
print('sim switch rate by decision round and run')
print(dec.groupby(['run', 'round_number']).sw.mean().unstack().round(3))
# by run: switch rate from each group
print(dec.groupby(['run', 'agent_group']).sw.mean().unstack().round(3))
