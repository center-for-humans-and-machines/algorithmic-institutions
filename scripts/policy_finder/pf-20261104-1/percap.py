"""Per-member contribution and punishment by manager and round, mixed pairings."""
import pandas as pd
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('zero_vs_ah')]
d['mgr'] = d.agent_group.map({0:'zero',1:'ah'})
d['valid'] = d.contribution_valid.astype(float)
t = d.groupby(['round_number','mgr']).agg(c=('contribution','mean'), p=('punishment','mean'), v=('valid','mean')).unstack()
print(t.round(2).to_string())
# contribution of movers: what do switchers contribute before moving
d = d.sort_values(['episode','participant_code','round_number'])
d['nextg'] = d.groupby(['episode','participant_code']).agent_group.shift(-1)
dec = d[d.round_number%4==3].copy(); dec['sw'] = dec.nextg != dec.agent_group
print(dec.groupby(['mgr','sw']).contribution.agg(['mean','size']).round(2))
