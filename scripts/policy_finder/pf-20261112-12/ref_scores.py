import pandas as pd
from aimanager.simulation.pool_scores import add_pool, against_anchor
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
add_pool(d)
s=against_anchor(d,'ah')
for m,f in s.items():
    print(m, (f.pool-f.other).mean().round(2), f.pool.mean().round(2), f.other.mean().round(2), f.members.mean().round(2), f.other_members.mean().round(2), len(f))
# per-round totals for zero_vs_ah
for r in d.run.unique():
    sub=d[d.run==r]
    g=sub.groupby(['episode','round_number','agent_group']).agg(pool=('pool','sum'),n=('pool','size'),c=('contribution','mean'),p=('punishment','mean')).reset_index()
    t=g.pivot_table(index=['episode','round_number'],columns='agent_group',values=['pool','n']).fillna(0)
    print(r); print(t.groupby('round_number').mean().round(1).T.to_string())
